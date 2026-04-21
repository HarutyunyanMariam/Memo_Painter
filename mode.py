# model.py

import os
import torch
import torch.nn as nn
from dataset import mydata
from torch.utils.data import DataLoader
import torch.optim as optim
from memory_network import Memory_Network
from generator import unet_generator
from discriminator import Discriminator
from util import zero_grad
from skimage.color import lab2rgb
import numpy as np
from PIL import Image


# =========================
# TRAIN
# =========================
def train(args):

    model_path = os.path.join(args.model_path, args.data_name)
    os.makedirs(model_path, exist_ok=True)

    log_path = os.path.join(model_path, f"{args.data_name}_train_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    tr_dataset = mydata(
        args.train_data_path,
        args.img_size,
        args.km_file_path,
        args.color_info
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    if args.test_with_train:
        te_dataset = mydata(
            args.test_data_path,
            args.img_size,
            args.km_file_path,
            args.color_info
        )

        te_loader = DataLoader(
            te_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    # Models
    mem = Memory_Network(
        args.mem_size,
        args.color_info,
        args.color_feat_dim,
        args.spatial_feat_dim,
        args.top_k,
        args.alpha
    ).to(device)

    generator = unet_generator(
        args.input_channel,
        args.output_channel,
        args.n_feats,
        args.color_feat_dim
    ).to(device)

    discriminator = Discriminator(
        args.input_channel + args.output_channel,
        args.color_feat_dim,
        args.img_size
    ).to(device)

    generator.train()
    discriminator.train()

    # Loss
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.SmoothL1Loss()

    # Optimizers
    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr*2, betas=(0.5,0.999))
    m_opt = optim.Adam(mem.parameters(), lr=args.lr)

    opts = [g_opt, d_opt, m_opt]

    # Resume
    start_epoch = 0
    best_loss = float("inf")

    if args.resume_epoch > 0:
        ckpt_path = os.path.join(model_path, f"checkpoint_{args.resume_epoch:03d}.pt")
        print(f"Resuming from {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)

        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        mem.load_state_dict(ckpt["memory"])

        mem.spatial_key = ckpt["mem_key"].to(device)
        mem.color_value = ckpt["mem_value"].to(device)
        mem.age = ckpt["mem_age"].to(device)
        mem.top_index = ckpt["mem_index"].to(device)

        g_opt.load_state_dict(ckpt["optimizer_g"])
        d_opt.load_state_dict(ckpt["optimizer_d"])
        m_opt.load_state_dict(ckpt["optimizer_m"])

        start_epoch = ckpt["epoch"] + 1

    # Training Loop
    for e in range(start_epoch, args.epoch):

        epoch_g = 0
        epoch_d = 0

        print(f"\nEpoch {e}")
        i = 0
        for batch in tr_loader:
            print(i)
            i+=1
            res_input = batch["res_input"].to(device)
            color_feat = batch["color_feat"].to(device)

            l_channel = (batch["l_channel"] / 100.0).to(device)
            ab_channel = (batch["ab_channel"] / 110.0).to(device)

            idx = batch["index"].to(device)
            bs = res_input.size(0)

            real_labels = torch.ones((bs,1)).to(device)
            fake_labels = torch.zeros((bs,1)).to(device)

            # Memory Training
            res_feature = mem(res_input)
            mem_loss = mem.unsupervised_loss(res_feature, color_feat, args.color_thres)

            zero_grad(opts)
            mem_loss.backward()
            m_opt.step()

            with torch.no_grad():
                res_feature = mem(res_input)
                mem.memory_update(res_feature, color_feat, args.color_thres, idx)

            # Discriminator
            dis_color_feat = color_feat.unsqueeze(2).unsqueeze(3)
            dis_color_feat = dis_color_feat.repeat(1,1,args.img_size,args.img_size)

            fake_ab = generator(l_channel, color_feat)

            real_out = discriminator(ab_channel, l_channel, dis_color_feat)
            fake_out = discriminator(fake_ab.detach(), l_channel, dis_color_feat)

            d_loss = criterion_GAN(real_out, real_labels) + \
                     criterion_GAN(fake_out, fake_labels)

            zero_grad(opts)
            d_loss.backward()
            d_opt.step()

            # Generator
            fake_ab = generator(l_channel, color_feat)
            fake_out = discriminator(fake_ab, l_channel, dis_color_feat)

            g_loss = criterion_GAN(fake_out, real_labels) + \
                     criterion_L1(fake_ab, ab_channel)

            zero_grad(opts)
            g_loss.backward()
            g_opt.step()

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()

        epoch_g /= len(tr_loader)
        epoch_d /= len(tr_loader)

        print(f"G: {epoch_g:.4f} | D: {epoch_d:.4f}")

        # log
        with open(log_path, "a") as f:
            f.write(f"Epoch {e} -> G: {epoch_g:.6f} D: {epoch_d:.6f}\n")

        # Save best model (SMALL FILE)
        if epoch_g < best_loss:

            best_loss = epoch_g

            torch.save({
                "generator": generator.state_dict(),
                "memory": mem.state_dict(),
                "mem_key": mem.spatial_key.cpu(),
                "mem_value": mem.color_value.cpu(),
                "mem_age": mem.age.cpu(),
                "mem_index": mem.top_index.cpu()
            }, os.path.join(model_path,"best_model.pt"))

            print("Best model saved")

        # Save checkpoint (LARGE FILE)
        if (e+1) % args.model_save_freq == 0:

            torch.save({
                "epoch":e,
                "generator":generator.state_dict(),
                "discriminator":discriminator.state_dict(),
                "memory":mem.state_dict(),
                "mem_key":mem.spatial_key.cpu(),
                "mem_value":mem.color_value.cpu(),
                "mem_age":mem.age.cpu(),
                "mem_index":mem.top_index.cpu(),
                "optimizer_g":g_opt.state_dict(),
                "optimizer_d":d_opt.state_dict(),
                "optimizer_m":m_opt.state_dict(),
            }, os.path.join(model_path,f"checkpoint_{e:03d}.pt"))

        # validation
        if args.test_with_train and (e+1)%args.test_freq==0:
            generator.eval()
            test_operation(args,generator,mem,te_loader,device,e)
            generator.train()





def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    test_dataset = mydata(
        img_path=args.test_data_path,
        img_size=args.img_size,
        km_file_path=args.km_file_path,
        color_info=args.color_info,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
   
    mem = Memory_Network(
        mem_size=args.mem_size,
        color_info=args.color_info,
        color_feat_dim=args.color_feat_dim,
        spatial_feat_dim=512,
        alpha=args.alpha,
    )
    generator = unet_generator(
        args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim
    )
   
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    mem.load_state_dict(ckpt["memory"])
    mem.spatial_key = ckpt["mem_key"].to(device)
    mem.color_value = ckpt["mem_value"].to(device)
    mem.age = ckpt["mem_age"].to(device)
    mem.top_index = ckpt["mem_index"].to(device)
   
    mem.to(device)
    generator.to(device).eval()
   
    test_operation(args, generator, mem, test_dataloader, device)








def test_operation(args, generator, mem, te_dataloader, device, e=-1):
    count = 0
    result_path = os.path.join(args.result_path, args.data_name)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
   
    with torch.no_grad():
        for i, batch in enumerate(te_dataloader):
            res_input = batch["res_input"].to(device)
            color_feat = batch["color_feat"].to(device)
            l_channel = (batch["l_channel"] / 255.0).to(device)
            ab_channel = (batch["ab_channel"] / 110.0).to(device)
           
            bs = res_input.size()[0]
           
            query = mem(res_input)
            top1_feature, _ = mem.topk_feature(query, 1)
            top1_feature = top1_feature[:, 0, :]
            result_ab_channel = generator(l_channel, top1_feature)
           
            real_image = torch.cat(
                [l_channel * 100, ab_channel * 110], dim=1
            ).cpu().numpy()
            fake_image = torch.cat(
                [l_channel * 100, result_ab_channel * 110], dim=1
            ).cpu().numpy()
            gray_image = torch.cat(
                [l_channel * 100, torch.zeros((bs, 2, args.img_size, args.img_size)).to(device)], dim=1
            ).cpu().numpy()
           
            all_img = np.concatenate([real_image, fake_image, gray_image], axis=2)
            all_img = np.transpose(all_img, (0, 2, 3, 1))
            rgb_imgs = [lab2rgb(ele) for ele in all_img]
            rgb_imgs = np.array(rgb_imgs)
            rgb_imgs = (rgb_imgs * 255.0).astype(np.uint8)
           
            for t in range(len(rgb_imgs)):
                if e > -1:
                    img = Image.fromarray(rgb_imgs[t])
                    name = f"{e:03d}_{count:04d}_result.png"
                    img.save(os.path.join(result_path, name))
                else:
                    name = "%04d_%s.png"
                    img = rgb_imgs[t]
                    h, w, c = img.shape
                    stride = h // 3
                    original = img[:stride, :, :]
                    Image.fromarray(original).save(
                        os.path.join(result_path, name % (count, "GT"))
                    )
                    result = img[stride : 2 * stride, :, :]
                    Image.fromarray(result).save(
                        os.path.join(result_path, name % (count, "result"))
                    )
                    if not args.test_only:
                        gray_img = img[2 * stride :, :, :]
                        Image.fromarray(gray_img).save(
                            os.path.join(result_path, name % (count, "gray"))
                        )
                count += 1
