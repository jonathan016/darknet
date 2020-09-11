import json
from random import Random, randint, random, sample as rand_sample, seed as rand_seed
from typing import Any, Dict
from warnings import warn

from imgaug.augmenters import Sequential, Resize as iaaResize
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
from numpy import array as np_array
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import Resize, Compose, ColorJitter, Grayscale, Normalize


class DetectionRecognitionDataset(Dataset):
    class FileLocations:
        def __init__(self, indices_path, coordinates_path):
            self.indices_path = indices_path
            self.coordinates_path = coordinates_path

    def __init__(self, root, file_locations: FileLocations, is_containing=True, seed=1, transform=None,
                 loader=default_loader):

        self.root = root
        self.file_locations = file_locations
        self.is_containing = is_containing

        self.data = self._load_data(seed)

        self.transform = transform
        self.loader = loader

    def _load_data(self, seed=1):
        coordinates = json.load(open(self.file_locations.coordinates_path, 'r'))['coordinates']
        coordinates = DataFrame.from_records(
            coordinates, columns=['class', 'shelf', 'frame', 'xleft', 'yupper', 'xright', 'ylower'])

        if self.is_containing:
            data = self._load_containing_frames(coordinates)
        else:
            data = self._load_non_containing_frames(seed)

        return data

    def _load_non_containing_frames(self, seed):
        data = []
        indices = json.load(open(self.file_locations.indices_path, 'r'))['without']
        Random(seed).shuffle(indices)

        for filename in indices:
            data.append((f'{filename}.jpg', []))

        return data

    def _load_containing_frames(self, coordinates):
        data = []
        indices = json.load(open(self.file_locations.indices_path, 'r'))['data']

        for class_data in indices:
            for filename in class_data[1]:
                shelf_frame = filename.split('-')
                shelf = int(shelf_frame[0].split('_')[1])
                frame = int(shelf_frame[1].split('frame')[1])
                image_values = coordinates[
                    (coordinates['shelf'] == shelf) & (coordinates['frame'] == frame)].values.tolist()
                for i, val in enumerate(image_values):
                    image_values[i] = [image_values[i][0], *image_values[i][3:]]
                data.append((f'{filename}.jpg', image_values))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, targets = self.data[index]
        image = self.loader(os.path.join(self.root, *filename.split('-')))

        if self.transform:
            image = self.transform(image)

        targets = self._format_yolo_targets(targets)
        return image, targets

    def _format_yolo_targets(self, targets):
        width_scale, height_scale = 720, 480
        targets = [[box[0], ((box[1] + box[3]) / 2) / width_scale, ((box[2] + box[4]) / 2) / height_scale,
                    (box[3] - box[1]) / width_scale, (box[4] - box[2]) / height_scale]
                   for box in targets]

        return targets


def _combine(identifiers, images, transform=None, min_resize=None, max_resize=None, random_seed=None):
    if random_seed:
        rand_seed(random_seed)

    if transform:
        if min_resize and max_resize:
            def resize_maintain_aspect_ratio(image):
                width, height = image.size
                aspect_ratio = width / height
                has_looped_for = 0
                while has_looped_for < 1000:
                    new_width, new_height = randint(min_resize, max_resize), randint(min_resize, max_resize)
                    if abs((new_width / new_height) - aspect_ratio) < .2:
                        break
                    has_looped_for += 1
                if has_looped_for == 1000:
                    random_scale = random() + randint(0, 3)
                    random_scale = random_scale if random_scale > .25 else .25
                    new_width, new_height = int(width * random_scale), int(height * random_scale)

                return Resize((new_height, new_width))(image)

            images = map(resize_maintain_aspect_ratio, images)
        images = [transform(image) for image in images]

    identifiers_images = list(zip(identifiers, images))
    backup_identifiers_images = list(zip(identifiers, images))

    combined_image_size = _get_combined_image_size(images)
    combined_image = Image.new('RGBA', combined_image_size)
    coordinates = {k: None for k, _ in identifiers_images}

    total_loop = 0
    while len(identifiers_images) != 0:
        # Total loop approaching 1000 means current combined image cannot fit all the images; a 'restart' is required.
        # This is to prevent infinite loop.
        if total_loop == 1000:
            identifiers_images = backup_identifiers_images
            combined_image_size = _get_combined_image_size(images)
            combined_image = Image.new('RGBA', combined_image_size)
            coordinates = {k: None for k, _ in identifiers_images}
            total_loop = 0

        for i in identifiers_images:
            identifier, image = i
            image_coordinate = _get_image_coordinate(image, combined_image_size)

            if _not_overlapping(coordinates, image_coordinate):
                combined_image.paste(image.convert('RGBA'), image_coordinate, image.convert('RGBA'))
                coordinates[identifier] = image_coordinate
                identifiers_images.remove(i)
        total_loop += 1

    return combined_image.convert('RGB'), coordinates


def _get_image_coordinate(image, base_image_size):
    image_width, image_height = image.size

    left, right = _get_safe_boundary(image_width, base_image_size[0])
    upper, lower = _get_safe_boundary(image_height, base_image_size[1])

    return left, upper, right, lower


def _get_safe_boundary(image_aspect, base_image_aspect):
    first = randint(0, base_image_aspect)
    second = first + image_aspect

    while second > base_image_aspect:
        first = randint(0, base_image_aspect)
        second = first + image_aspect

    return first, second


def _not_overlapping(coordinates, image_coordinate) -> bool:
    no_overlap = [_get_intersection(coordinate, image_coordinate) == 0.0 for coordinate in coordinates.values()]

    return all(no_overlap)


def _get_intersection(box1, box2):
    if box1 is None:
        return 0

    x = min(box1[0], box2[0]), max(box1[2], box2[2])
    y = min(box1[1], box2[1]), max(box1[3], box2[3])

    box1 = box1[2] - box1[0], box1[3] - box1[1]
    box2 = box2[2] - box2[0], box2[3] - box2[1]

    raw_union_width = x[1] - x[0]
    raw_union_height = y[1] - y[0]
    intersection_width = box1[0] + box2[0] - raw_union_width
    intersection_height = box1[1] + box2[1] - raw_union_height

    intersection = intersection_width * intersection_height if intersection_width > 0 and intersection_height > 0 else 0
    return float(intersection)


def _get_combined_image_size(images):
    sum_width = sum([x.size[0] for x in images])
    sum_height = sum([x.size[1] for x in images])

    max_resolution = max(
        sum_width + randint(sum_width // 2, sum_width), sum_height + randint(sum_height // 2, sum_height))
    combined_image_size = (max_resolution, max_resolution)

    return combined_image_size


def _combine_from_path(source_labels, transform=None, min_resize=None, max_resize=None, random_seed=None):
    images = [Image.open(path) for path in source_labels.values()]
    identifiers = [identifier for identifier in source_labels.keys()]

    return _combine(identifiers, images, transform, min_resize, max_resize, random_seed)


def combined_zoom_out(source: Dict[Any, str] = None, identifiers: list = None, images: list = None,
                      individual_transform=None, min_resize: int = None, max_resize: int = None, seed: int = None):
    """The proposed augmentation technique in this project's proposal. ``source`` is mutually exclusive to
    ``identifiers`` and ``images``, and vice versa.

    The idea is to accept a list of images, where each image contains only an object, and then combined to a single
    image to simulate the typical object detection input. A transformation can be applied for each image,
    with addition of randomized resizing. This method also generates and returns the coordinates for each image in
    the combined image to be used for further processing.

    :param source: A dictionary where the key is an identifier of any hashable type and value of image path (string).
    :param identifiers: Identifiers of ``images`` to get bounding boxes of the corresponding image after augmentation.
    :param images: List of ``PIL Image``s, each is ordered according to the order of ``identifiers``.
    :param individual_transform: Transformations to be applied to each image in ``images``. Must be from
      ``torchvision.transforms`` module and ideally should not contain any color changing transformation. For color
      changing transformation, set them on the ``combined_transform`` parameter.
    :param min_resize: Minimum resized image's width and/or height.
    :param max_resize: Maximum resized image's width and/or height.
    :param seed: Seed for the randomized coordinate and image resizing value randomization.
    :return: A tuple of combined image in RGB format and a coordinate dictionary, where the key is the specified
      identifiers and the value is a tuple of (left, upper, right, lower) coordinate of each image's bounding box in
      the combined image.
    """

    if source and (identifiers or images):
        raise ValueError('source must be defined without identifiers and images')
    if (identifiers and images) and source:
        raise ValueError('identifiers and images must be defined without source')

    __check_for_color_changing_transform(individual_transform)

    combined_image, bounding_boxes = None, None
    if source:
        combined_image, bounding_boxes = _combine_from_path(source, individual_transform, min_resize, max_resize, seed)
    elif identifiers and images:
        assert len(identifiers) == len(images)
        combined_image, bounding_boxes = _combine(identifiers, images, individual_transform, min_resize, max_resize,
                                                  seed)

    return combined_image, bounding_boxes


def __check_for_color_changing_transform(individual_transform):
    color_changing_transformations = [ColorJitter, Grayscale, Normalize]

    if type(individual_transform) is Compose:
        if any([type(t) in color_changing_transformations for t in individual_transform.transforms]):
            warn('Individual image transform should not contain color changing transformation(s)')
    elif type(individual_transform) in color_changing_transformations:
        warn('Individual image transform should not contain color changing transformation(s)')


class GroZiDetectionDataset(ImageFolder):
    """Encapsulates GroZi-120 dataset with combined zoom out augmentation as proposed in this project's proposal.

    This dataset wrapper is usable for SSD and YOLO (v2 and v3) models only for now. After selecting
    ``max_data_in_combined`` individual product images, the selected images are combined with combined zoom out
    augmentation and later formatted to this dataset's specified model's format.

    Arguments:
        root (string): Root folder of the individual product images.
        transform (callable, optional): Transformations from ``torchvision.transforms`` module to be applied to each
            individual product images prior to combining.
        target_transform (callable, optional): Not directly used, but will be used by ``ImageFolder``'s
            implementation to transform the target values of the dataset.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an ``Image`` file
            and check if the file is a valid_file (used to check of corrupt files).
        transform_to_tensor (bool, optional): A flag to denote whether the combined image should be converted to
            tensor or not.
        min_resize (int, optional): The minimum resize width/height of each individual product image.
        max_resize (int, optional): The maximum resize width/height of each individual product image.
        max_data_usage (int, optional): Maximum individual product image usage in the combined zoom out augmentation
            technique.
        max_data_in_combined (int, optional): Maximum present individual product image in the combined image.
        max_object (int, optional): Number of possible objects in the dataset. Only to be specified if ``model`` is
            ``GroZiDetectionDataset.YOLO``.
        seen_images (int, optional): Number of seen images of the YOLO model. Useful for changing image size (
            multi-scale training) for YOLO models. Only to be specified if ``model`` is ``GroZiDetectionDataset.YOLO``.
        batch_size (int, optional): Batch size of training data. Useful for changing image size (
            multi-scale training) for YOLO models. Only to be specified if ``model`` is ``GroZiDetectionDataset.YOLO``.
    """
    SSD: str = 'SSD'
    YOLO: str = 'YOLO'

    def __init__(self, root, model: str = SSD, transform=Compose([]), target_transform=None, loader=default_loader,
                 is_valid_file=None, transform_to_tensor: bool = True, min_resize: int = None, max_resize: int = None,
                 max_data_usage: int = 1, max_data_in_combined: int = 1, max_object: int = None, seen_images: int =
                 0, batch_size: int = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        assert model == self.YOLO or model == self.SSD
        self.model = model
        if self.model == self.YOLO:
            assert max_object is not None and max_object > 0
            assert seen_images is not None and seen_images >= 0
            assert batch_size is not None and batch_size > 0
            self.max_object = max_object
            self.seen_images = seen_images
            self.batch_size = batch_size

        self.max_data_usage = max_data_usage
        self.max_data_in_combined = max_data_in_combined
        self.kwargs = {
            'min_resize': min_resize,
            'max_resize': max_resize,
            'individual_transform': self.transform
        }

        self.transform_to_tensor = transform_to_tensor
        self.usage = {k: 0 for k in self.samples}

    def __len__(self):
        return len(self.samples) * self.max_data_usage // self.max_data_in_combined

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError

        used_samples = self.__get_used_samples()

        images = []
        index_labels = []
        for index, sample in enumerate(used_samples):
            path, label = sample
            images.append(self.loader(path))
            # index is added since bounding_boxes keys will have overlapping values due to same label being in the
            # same image if index is not added. By grouping index and label to a tuple, a unique key is guaranteed
            # and no overlapping value in bounding_boxes dictionary is possible even with products of the same label
            # occurring more than once in the combined_image.
            index_labels.append((index, label))
            self.usage[(path, label)] += 1

        combined_image, bounding_boxes = combined_zoom_out(identifiers=index_labels, images=images, **self.kwargs)

        image, bounding_boxes = self.resize_image_and_bounding_boxes(combined_image, bounding_boxes)

        labels = [label for _, label in bounding_boxes.keys()]
        bounding_boxes = list(bounding_boxes.values())

        return self._format_yolo(image, labels, bounding_boxes)

    @staticmethod
    def resize_image_and_bounding_boxes(combined_image, bounding_boxes):
        combined_image = np_array(combined_image)
        bounding_boxes = BoundingBoxesOnImage([
            BoundingBox(*coordinates, label=identifier) for identifier, coordinates in bounding_boxes.items()
        ], shape=combined_image.shape)

        combined_image, bounding_boxes = Sequential([iaaResize((416, 416))])(
            image=combined_image, bounding_boxes=bounding_boxes)

        combined_image = Image.fromarray(combined_image)
        bounding_boxes = {b.label: (b.x1_int, b.y1_int, b.x2_int, b.y2_int) for b in bounding_boxes}

        return combined_image, bounding_boxes

    @staticmethod
    def _format_yolo(image, labels, bounding_boxes):
        assert len(labels) == len(bounding_boxes)

        yolo_labels = []
        for i in range(len(labels)):
            cx, cy, w, h = GroZiDetectionDataset._get_center_size_coordinates(bounding_boxes[i], image.size)
            yolo_labels.append([int(labels[i]), cx, cy, w, h])

        return image, yolo_labels

    @staticmethod
    def _get_center_size_coordinates(bounding_boxes, image_size):
        x_left, y_upper, x_right, y_lower = bounding_boxes
        image_width, image_height = image_size

        cx = (((x_right - x_left) / 2) + x_left) / image_width
        cy = (((y_lower - y_upper) / 2) + y_upper) / image_height
        w = (x_right - x_left) / image_width
        h = (y_lower - y_upper) / image_height

        return cx, cy, w, h

    def __get_used_samples(self):
        indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
        used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
        fully_used = [self.usage[k] == self.max_data_usage for k in used_samples]

        while all(fully_used):
            indexes = rand_sample(range(0, len(self.samples)), randint(1, self.max_data_in_combined))
            used_samples = [k for i, k in enumerate(self.samples) if i in indexes]
            fully_used = [self.usage[k] == 20 for k in used_samples]

        return used_samples


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--root', default='~/rpdr-config-results/data/cropped/')
    parser.add_argument('--frames_root', default='~/rpdr-config-results/data/frames/')
    parser.add_argument('--output_dir', default='data/obj/')
    parser.add_argument('--train_list', default='data/train.txt')
    parser.add_argument('--val_list', default='data/valid.txt')
    parser.add_argument('--test_list', default='data/test.txt')
    parser.add_argument('--ic_list', default='indices_coordinates/')
    parser.add_argument('--min_resize', default=70, type=int)
    parser.add_argument('--max_resize', default=250, type=int)
    parser.add_argument('--max_usage', default=300, type=int)
    parser.add_argument('--max_combined', default=10, type=int)

    args = parser.parse_args()

    dataset = GroZiDetectionDataset(
        args.root, GroZiDetectionDataset.YOLO, min_resize=args.min_resize, max_resize=args.max_resize,
        max_data_usage=args.max_usage, max_data_in_combined=args.max_combined, max_object=15, batch_size=1)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_indices = open(args.train_list, 'a')
    index = 1
    for data in dataset:
        image, targets = data

        image.save(os.path.join(args.output_dir, f'{index:08d}.jpg'))
        file = open(os.path.join(args.output_dir, f'{index:08d}.txt'), 'a')
        for target in targets:
            file.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n')
        train_indices.write(f"{os.path.join(args.output_dir, f'{index:08d}.jpg')}\n")
        index += 1

    val_containing_dataset = DetectionRecognitionDataset(args.frames_root, DetectionRecognitionDataset.FileLocations(
        os.path.join(args.ic_list, 'detect_val_frames-containing.json'),
        os.path.join(args.ic_list, 'detect_val_test_coordinates.json')
    ), is_containing=True)
    val_not_containing_dataset = DetectionRecognitionDataset(
        args.frames_root, DetectionRecognitionDataset.FileLocations(
            os.path.join(args.ic_list, 'detect_val_frames-not_containing.json'),
            os.path.join(args.ic_list, 'detect_val_test_coordinates.json')
        ), is_containing=False)

    val_indices = open(args.val_list, 'a')
    for data in val_containing_dataset:
        image, targets = data

        image.save(os.path.join(args.output_dir, f'{index:08d}.jpg'))
        file = open(os.path.join(args.output_dir, f'{index:08d}.txt'), 'a')
        for target in targets:
            file.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n')
        val_indices.write(f"{os.path.join(args.output_dir, f'{index:08d}.jpg')}\n")
        index += 1

    for data in val_not_containing_dataset:
        image, targets = data

        image.save(os.path.join(args.output_dir, f'{index:08d}.jpg'))
        file = open(os.path.join(args.output_dir, f'{index:08d}.txt'), 'a')
        for target in targets:
            file.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n')
        val_indices.write(f"{os.path.join(args.output_dir, f'{index:08d}.jpg')}\n")
        index += 1

    test_containing_dataset = DetectionRecognitionDataset(args.frames_root, DetectionRecognitionDataset.FileLocations(
        os.path.join(args.ic_list, 'detect_test_frames-containing.json'),
        os.path.join(args.ic_list, 'detect_val_test_coordinates.json')
    ), is_containing=True)
    test_not_containing_dataset = DetectionRecognitionDataset(
        args.frames_root, DetectionRecognitionDataset.FileLocations(
            os.path.join(args.ic_list, 'detect_test_frames-not_containing.json'),
            os.path.join(args.ic_list, 'detect_val_test_coordinates.json')
        ), is_containing=False)

    test_indices = open(args.test_list, 'a')
    for data in test_containing_dataset:
        image, targets = data

        image.save(os.path.join(args.output_dir, f'{index:08d}.jpg'))
        file = open(os.path.join(args.output_dir, f'{index:08d}.txt'), 'a')
        for target in targets:
            file.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n')
        test_indices.write(f"{os.path.join(args.output_dir, f'{index:08d}.jpg')}\n")
        index += 1

    for data in test_not_containing_dataset:
        image, targets = data

        image.save(os.path.join(args.output_dir, f'{index:08d}.jpg'))
        file = open(os.path.join(args.output_dir, f'{index:08d}.txt'), 'a')
        for target in targets:
            file.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]}\n')
        test_indices.write(f"{os.path.join(args.output_dir, f'{index:08d}.jpg')}\n")
        index += 1
