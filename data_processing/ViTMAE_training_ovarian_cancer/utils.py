import torch
import random
import numpy as np
import anndata as ad
from tqdm import tqdm
from skimage.transform import resize
import torchvision.transforms.functional as F
from scipy.spatial.distance import pdist, squareform
from transformers import ViTImageProcessor, ConvNextModel

def vit_resize(
    image: np.ndarray,
    size=(224,224), # tuple: (224,224)
) -> np.ndarray:
    resized_image = np.array([resize(channel, size, anti_aliasing=True) for channel in image])
    return resized_image

class CustomViTImageProcessor(ViTImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def resize(
        self,
        image: np.ndarray,
        size,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return vit_resize(
            image,
            size=output_size
        )
    

def m_percent_of_smallest_value(data, m):
    """
    if data is a dict, it returns m percent of the smallest value in the dict.
    if data is a number, it returns m percent of that number.
    """
    try:
        smallest_value = min(data.values())
    except:
        smallest_value = data
    m_percent_value = ((m / 100) * smallest_value) // 1 # round down
    return m_percent_value

def split_indices(total_samples: int, pct_val: float, pct_test: float, seed: int = 42):
    """
    Split indices into training, validation, and test sets.

    Args:
        total_samples (int): Total number of samples.
        pct_val (float): Percentage of data to use for validation (e.g., 5 for 5%).
        pct_test (float): Percentage of data to use for testing (e.g., 10 for 10%).
        seed (int): Random seed for reproducibility.

    Returns:
        train_indices (np.ndarray): Array of training indices.
        val_indices (np.ndarray): Array of validation indices.
        test_indices (np.ndarray): Array of test indices.
    """
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)

    n_val = int(total_samples * pct_val / 100)
    n_test = int(total_samples * pct_test / 100)
    n_train = total_samples - n_val - n_test

    val_indices = indices[:n_val]
    test_indices = indices[n_val:n_val + n_test]
    train_indices = indices[n_val + n_test:]

    return train_indices, val_indices, test_indices

def clip_image(in_tensor):
    return  np.clip(in_tensor, 0, 255)

def multiply_channel(in_tensor):
    out = torch.zeros((3, 1 ,1)) + in_tensor
    return out

# def feature_extractor_custom(in_tensor):
#     image_processor = CustomViTImageProcessor(do_rescale=False, do_normalize=False, do_resize=True)
#     out = image_processor(in_tensor[0], input_data_format="channels_first")
#     out = out["pixel_values"][0]
#     return out

def feature_extractor(in_tensor):
    image_processor = ViTImageProcessor(do_rescale=False, do_normalize=False, do_resize=False)
    out = image_processor(in_tensor)
    out = out["pixel_values"][0]
    return out

class RescaleTo01:
    def __call__(self, img):
        img_min, img_max = img.min(), img.max()
        return (img - img_min) / (img_max - img_min)
        
class Random90DegreeRotation:
    def __call__(self, img):
        # Randomly choose one of the 90-degree rotations
        degrees = random.choice([0, 90, 180, 270])
        return F.rotate(img, degrees)

def filter_by_unique_id(data, cell_ids, unique_id):
    """
    Args:
    data (torch.Tensor): A tensor of shape (14560, 2048)
    cell_ids (torch.Tensor): A tensor of shape (14560)
    unique_id (int): The unique id to filter the data by
    
    Returns:
    torch.Tensor: The filtered values from the data corresponding to the unique id
    """
    # Ensure cell_ids is a 1D tensor
    if isinstance(cell_ids, torch.Tensor):
        cell_ids = cell_ids.view(-1)
    else:
        cell_ids = np.ravel(cell_ids)
    # Create a mask to filter the data
    mask = cell_ids == unique_id
    # Apply the mask to the data
    filtered_data = np.array(data[mask])
    return filtered_data


def get_anndata_obj(output_avg, targets, cell_ids, channels=None):
    fv_list = []
    labels_list = []
    channels_list = []
    print("Number of unique cells",len(np.unique(cell_ids)))
    for unique_ids in np.unique(cell_ids):
        filtered_values = filter_by_unique_id(output_avg, cell_ids, unique_ids)
        fv_list.append(filtered_values.reshape(-1))
        filtered_labels = filter_by_unique_id(targets, cell_ids, unique_ids)
        labels_list.append(filtered_labels[0].item())
        if channels is not None:
            filtered_channels = filter_by_unique_id(channels, cell_ids, unique_ids)
            channels_list.append(filtered_channels[0].item())
        # pdb.set_trace()
    concat_output = np.array(fv_list)
    concat_labels = np.array(labels_list)
    concat_channels = np.array(channels_list)
    data = ad.AnnData(concat_output)
    data.obs["targets"] = concat_labels
    if channels is not None:
        data.obs["channels"] = concat_channels # "channels" make no sense after concetenation of channels to one cell by filter_by_unique_id.
    data.obs["cell_ids"] = np.unique(cell_ids)
    return data

def get_cosine_similarity(latent_vars):
    """
    Args.
        latent_vars (np.ndarray): 2 dim array of n_cells x (latent dim. times number channels)
    Returns.
        cosine_sim (array): matrix of n_cells x n_cells with corresponding pairwise cosine similarities.
    """
    # Compute cosine distance and then convert it to similarity 
    cosine_dist = pdist(latent_vars, 'cosine') 
    cosine_sim = 1 - squareform(cosine_dist)
    return cosine_sim

def get_outputs(datloader, model, device, layer_outputs):
    channel_return_flag = False
    with torch.no_grad():
        data_iter = iter(datloader)
        try:
            images, targets, cell_ids, channels = next(data_iter)
            channel_return_flag = True
        except:
            images, targets, cell_ids = next(data_iter)
        images = images.to(device)
        # pdb.set_trace()
        o = model(images)
        output_avg = torch.mean(a = o.hidden_states[-1].cpu().detach(), axis = 1)
        # output_avg = (torch.mean(layer_outputs['encoder_layernorm'], axis = 1).detach().cpu().numpy())
        
        for i in tqdm((range(len(datloader) - 2))):
            if channel_return_flag:
                images, t, c_id, chan_id = next(data_iter)
                images = images.to(device)
                o = model(images)
                output_avg = torch.cat((output_avg, torch.mean(a = layer_outputs['encoder_layernorm'].cpu().detach(), axis = 1)), 0)
                targets = torch.cat((targets, t), 0)
                cell_ids = torch.cat((cell_ids, c_id), 0)
                channels = torch.cat((channels, chan_id), 0)
            else:
                images, t, c_id = next(data_iter)
                images = images.to(device)
                o = model(images)
                # pdb.set_trace()
                output_avg = torch.cat((output_avg, torch.mean(a = layer_outputs['encoder_layernorm'].cpu().detach(), axis = 1)), 0)
                targets = torch.cat((targets, t), 0)
                cell_ids = torch.cat((cell_ids, c_id), 0)
        try:
            return output_avg, targets, cell_ids, channels
        except:
            return output_avg, targets, cell_ids

def inference(model, dataloader, layer_outputs, device=None):
    """
    Run inference on a dataset using a ViT-style model and extract embeddings.

    This function performs a forward pass of the given model over all samples
    in the `dataloader`, and collects:
      - CLS token embeddings
      - Mean-pooled patch embeddings
      - Corresponding labels
      - Cell identifiers

    The model is run in evaluation mode with `mask_ratio = 0` to ensure no masking.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to run inference with. Must have a `.config.mask_ratio` attribute.
    
    dataloader : torch.utils.data.DataLoader
        A DataLoader providing batches of input images, labels, and cell IDs.
        Each batch must return: (images, labels, ids).

    layer_outputs : dict
        A dictionary that holds intermediate layer outputs. Must contain key 'encoder_layernorm',
        which stores the output tensor after model(imgs) forward pass.

    device : torch.device or None, optional
        Device on which to run inference. If None, defaults to CUDA if available, else CPU.

    Returns
    -------
    cls_outputs : np.ndarray
        Array of CLS token embeddings, shape (N, 1, D).

    pooled_patch_outputs : np.ndarray
        Array of mean-pooled patch embeddings (excluding CLS token), shape (N, D).

    out_labels : np.ndarray
        Array of labels corresponding to each sample, shape (N,).

    out_cell_ids : np.ndarray
        Array of cell IDs corresponding to each sample, shape (N,).
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.config.mask_ratio = 0
    model.eval()
    cls_outputs = []
    pooled_patch_outputs = []
    out_labels = []
    cell_ids = []
    with torch.no_grad():
        for imgs, labels, ids in tqdm(dataloader):
            imgs = imgs.to(device)
            model(imgs)
            out = layer_outputs['encoder_layernorm']
            cls = out[:, :1, :]
            cls_outputs.append(cls.cpu().numpy())
            pool = torch.mean(out[:, 1:, :], dim=1)
            pooled_patch_outputs.append(pool.cpu().numpy())
            out_labels.append(np.array(labels))
            cell_ids.append(np.array(ids))

    cls_outputs = np.concatenate(cls_outputs)
    pooled_patch_outputs = np.concatenate(pooled_patch_outputs)
    out_labels = np.concatenate(out_labels)
    out_cell_ids = np.concatenate(cell_ids)
    return cls_outputs, pooled_patch_outputs, out_labels, out_cell_ids