import tonic
from torchvision import transforms

if __name__ == '__main__':
    name_list = ["indoor_flying", "outdoor_day"]

    transform = transforms.Compose([
        tonic.transforms.ToTimesurface(sensor_size=tonic.datasets.MVSEC.sensor_size), ])

    for i in range(2):
        dataset = tonic.datasets.MVSEC(save_to='/home/hexiang/data/datasets/', scene=name_list[i], transform=transform)
        x = dataset[0]