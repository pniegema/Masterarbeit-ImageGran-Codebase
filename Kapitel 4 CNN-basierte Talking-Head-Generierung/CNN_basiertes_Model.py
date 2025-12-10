import torch
import torch.nn as nn
import torchvision.models as models
from util import count_parameters
from matplotlib import pyplot as plt
from GRID import GRIDDataset, get_landmarks
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomLoss(nn.Module):
    def __init__(self, lambda_l1=1, lambda_vgg=0.05, lambda_mouth=1):
        super(CustomLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.curr_l1 = 0
        self.vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)  # Use first 16 layers of VGG16
        self.curr_vgg = 0
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.curr_mouth = 0
        self.lambda_mouth = lambda_mouth

    def normalize_vgg_input(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def perceptual_loss(self, x, y):
        loss = 0
        layers = [4, 9, 16]  # Relu1_2, Relu2_2, Relu3_3
        x = self.normalize_vgg_input(x)
        y = self.normalize_vgg_input(y)

        for layer in layers:
            x_features = self.vgg[:layer](x)
            y_features = self.vgg[:layer](y)
            loss += self.L1(x_features, y_features)
        return loss

    def forward(self, output, target):
        l1_loss = self.L1(output, target)
        perceptual_loss = self.perceptual_loss(output, target)

        #Mouth Rectangle (39,84), (73,104)
        mouth_target = target[:, :, 84:105, 39:74]
        mouth_output = output[:, :, 84:105, 39:74]

        mouth_loss = self.L1(mouth_output, mouth_target)


        self.curr_mouth = mouth_loss.item() * self.lambda_mouth
        self.curr_l1 = l1_loss.item() * self.lambda_l1
        self.curr_vgg = perceptual_loss.item() * self.lambda_vgg


        return l1_loss * self.lambda_l1 + perceptual_loss * self.lambda_vgg + mouth_loss * self.lambda_mouth



class IdentityEncoder(nn.Module):
    """
    Input: Image 3x112x112
    Output: Latent representation of the identity with Standard 256 dimensions

    """
    def __init__(self, latent_dim=256):
        super(IdentityEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.identity_latent = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )


        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv2_5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc6_7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*5*5, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_5(x)
        x = self.pool1(x)
        skip1 = x
        x = self.conv2(x)
        skip2 = x
        x = self.conv2_5(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6_7(x)
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        self.identity_latent = x

        return x, [skip1, skip2]

class FrameDecoder(nn.Module):
    """
    Input: Latent representation of the identity + audio latent (256 + 256 = 512)
    Output: Generated Frame 3x112x112

    """
    def __init__(self, latent_dim=512):
        super(FrameDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU()
        )


        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 96, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(96 + 256, 96, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(96 + 96, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.unblur1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.tconv7 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=1),
            nn.ReLU(),
        )

        self.unblur2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )


    def forward(self, x, SkipConnections):
        x = self.fc1(x)
        x = x.view(-1, 512, 1, 1)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(torch.cat((x, SkipConnections[1]), dim=1))
        x = self.tconv6(torch.cat((x, SkipConnections[0]), dim=1))
        x = self.unblur1(x)
        x = self.tconv7(x)
        x = self.unblur2(x)
        return x

class AudioEncoder(nn.Module):
    """
    Input: Audio MFCC 1x12x21

    Output: Latent representation of the audio with Standard 256 dimensions
    """
    def __init__(self, latent_dim=256):
        super(AudioEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.audio_latent = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc6_7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.fc6_7(x)
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)

        self.audio_latent = x

        return x


class Speech2Vid(nn.Module):
    def __init__(self, identity_encoder_path = None, frame_decoder_path = None, audio_encoder_path = None):
        super(Speech2Vid, self).__init__()
        if identity_encoder_path is not None:
            self.identity_encoder = IdentityEncoder(256)
            self.identity_encoder.load_state_dict(torch.load(identity_encoder_path))
        else:
            self.identity_encoder = IdentityEncoder(256)

        if frame_decoder_path is not None:
            self.frame_decoder = FrameDecoder(522)
            self.frame_decoder.load_state_dict(torch.load(frame_decoder_path))
        else:
            self.frame_decoder = FrameDecoder(522)

        if audio_encoder_path is not None:
            self.audio_encoder = AudioEncoder(256)
            self.audio_encoder.load_state_dict(torch.load(audio_encoder_path))
        else:
            self.audio_encoder = AudioEncoder(256)

        self.noise_generator = nn.GRU(10, 10, 2, batch_first=True).to(device)
        self.audio_gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=False)
        self.combined_latent = None

    def forward(self, audio, image):
        identity_latent, skip_connections = self.identity_encoder(image)

        noise_latent = torch.randn(audio.shape[0], 10)
        noise_latent = noise_latent.to(device)
        noise_latent = noise_latent.unsqueeze(1)  # Reshape to (batch, seq_len=1, 10)
        _, noise_latent = self.noise_generator(noise_latent)  # Output shape: (num_layers, batch, 10)
        noise_latent = noise_latent[-1]
        #Only positive values
        noise_latent = F.relu(noise_latent)





        audio_latent = self.audio_encoder(audio)


        self.combined_latent = torch.cat((identity_latent, audio_latent, noise_latent), 1)

        image = self.frame_decoder(torch.cat((identity_latent, audio_latent, noise_latent), 1), skip_connections)

        return image


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler=None):
    l1_loss = []
    vgg_loss = []
    mouth_loss = []


    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for i, (audio, identity_frame, training_frame) in enumerate(train_loader):
            audio = audio.to(device)
            identity_frame = identity_frame.to(device)
            training_frame = training_frame.to(device)

            optimizer.zero_grad()
            output = model(audio, identity_frame)
            loss = criterion(output, training_frame)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i+1) % 10 == 0:
                l1_loss.append(criterion.curr_l1)
                vgg_loss.append(criterion.curr_vgg)
                mouth_loss.append(criterion.curr_mouth)


            if (i+1) % 200 == 0:
                print(f"Epoch {epoch +1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")

            if (i+1) % 600 == 0:
                # Plot a sample audio latent
                audio_latent = model.audio_encoder.audio_latent
                identity_latent = model.identity_encoder.identity_latent
                combined_latent = model.combined_latent
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.plot(audio_latent[0].cpu().detach().numpy())
                plt.title("Audio Latent")
                plt.subplot(1, 3, 2)
                plt.plot(identity_latent[0].cpu().detach().numpy())
                plt.title("Identity Latent")
                plt.subplot(1,3,3)
                plt.plot(combined_latent[0].cpu().detach().numpy())
                plt.title("Combined Latent")
                plt.show()

                # Plot the different losses in one plot
                plt.figure(figsize=(10, 5))
                plt.plot(l1_loss, label="L1 Loss")
                plt.plot(vgg_loss, label="VGG Loss")
                plt.plot(mouth_loss, label="Mouth Loss")
                plt.legend()
                plt.title("Losses")
                plt.show()

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(identity_frame[0].permute(1, 2, 0).cpu().detach().numpy())
                plt.title("Identity Frame")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(output[0].permute(1, 2, 0).cpu().detach().numpy())
                plt.title("Generated Frame")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(training_frame[0].permute(1, 2, 0).cpu().detach().numpy())
                plt.title("Training Frame")
                plt.axis("off")
                plt.show()


        train_loss /= len(train_loader)
        print(f"Epoch {epoch +1}/{num_epochs}, Loss: {train_loss}")
        end = time.time()
        print(f"Time taken for epoch: {end - start_time}")


        if scheduler is not None:
            scheduler.step()

        #temporarly Save the model every epoch
        torch.save(model.state_dict(), "Speech2Vid_temp.pth")

    return model

def plot_model(model, dataloader, num_batches=2, batch_size=8):
    """"
    Plot the model output for a few samples from the dataloader
    """

    for i in range(num_batches):
        audio, identity_frame, training_frame = next(iter(dataloader))
        audio = audio.to(device)
        identity_frame = identity_frame.to(device)
        training_frame = training_frame.to(device)

        output = model(audio, identity_frame)

        for j in range(batch_size):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(identity_frame[j].permute(1, 2, 0).cpu().detach().numpy())
            plt.title("Identity Frame")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(output[j].permute(1, 2, 0).cpu().detach().numpy())
            plt.title("Generated Frame")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(training_frame[j].permute(1, 2, 0).cpu().detach().numpy())
            plt.title("Training Frame")
            plt.axis("off")
            plt.show()

def main():

    gen_model = Speech2Vid().to(device)
    gen_model.load_state_dict(torch.load("YourModelPath.pth"))


    count_parameters(gen_model)

    root_dir = r"YourDatasetPath"
    dataset = GRIDDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    epochs = 20
    gen_criterion = CustomLoss(lambda_l1=1.5, lambda_vgg=0.05, lambda_mouth=2)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.003)

    #Optional Scheduler
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(dataloader), epochs=epochs)

    gen_model = train(gen_model, dataloader, None, epochs, gen_criterion, gen_optimizer, None)

    torch.save(gen_model.state_dict(), "YourSavePath.pth")


    plot_model(gen_model, dataloader, num_batches=2, batch_size=8)




if __name__ == "__main__":
    main()