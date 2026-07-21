"""Configuration module for VisTA-SR training and evaluation."""

import torch

class Config:
    def __init__(self, mode: str = "train_srgan", exp_name: str = "vista_sr_experiment"):
        # General configuration
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upscale_factor = 4
        self.mode = mode
        self.exp_name = exp_name

        # Dataset & Training hyperparameters
        self.train_image_dir = "/home/lion397/data/datasets/GEMINI/Training_All_221201/train"
        self.valid_image_dir = "/home/lion397/data/datasets/GEMINI/Training_All_221201/val"

        self.image_size = 256
        self.d_image_size = 96
        self.stn_image_size = self.d_image_size
        self.batch_size = 8
        self.num_workers = 4

        # Checkpoints
        self.resume = False
        self.start_epoch = 0
        self.epochs = 200
        self.resume_d_weight = f"results/{exp_name}/d-last.pth"
        self.resume_g_weight = f"results/{exp_name}/g-last.pth"

        # Loss weights
        self.pixel_weight = 1.0
        self.content_weight = 1.0
        self.adversarial_weight = 0.004
        self.similarity_weight = 1.0

        # Optimizer parameters
        self.d_model_lr = 1e-4
        self.g_model_lr = 1e-4
        self.d_model_betas = (0.9, 0.999)
        self.g_model_betas = (0.9, 0.999)

        # Scheduler parameters
        self.d_scheduler_step_size = 200
        self.g_scheduler_step_size = 200
        self.d_scheduler_gamma = 0.1
        self.g_scheduler_gamma = 0.1

        self.print_frequency = 50
