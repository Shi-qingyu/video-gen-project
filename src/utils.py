import torch
from transformers import T5EncoderModel, T5Tokenizer


def prepare_attention_mask(
    prompt,
    words,
    bboxes,
    tokenizer: T5Tokenizer,
    height: int,
    width: int,
    num_frames: int,
):
    attention_mask = torch.zeros(
        size=(226 + num_frames * height * width, 226 + num_frames * height * width)
    )

    word_ids, word_lens = prepare_word_ids(
        prompt,
        words,
        tokenizer
    )
    
    for word_idx, word_len, bbox in zip(word_ids, word_lens, bboxes):
        video_mask = torch.zeros(size=(num_frames, height, width))
        bbox[::2] = int(bbox[::2] * width)
        bbox[1::2] = int(bbox[1::2] * height)
        video_mask[:, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        video_mask = video_mask.flatten()[None].repeat(word_len, 1)

        attention_mask[226:, word_idx: word_idx+word_len] = video_mask.transpose(0, 1)
        attention_mask[word_idx: word_idx+word_len, 226:] = video_mask

    attention_mask[226:, :226] = torch.where(attention_mask[226:, :226], 9, -1e8)
    attention_mask[:226, 226:] = torch.where(attention_mask[:226, 226:], 9, -1e8)

    return attention_mask


def prepare_word_ids(
    prompt,
    words,
    tokenizer: T5Tokenizer,
):
    input_ids = tokenizer(
        prompt,
        max_length=226,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids[0] # (226, )

    word_ids = []
    word_lens = []

    for word in words:
        word_idx = tokenizer(
            word,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids[0]

        word_len = len(word_idx)
        word_lens.append(word_len)
        for idx in range(len(input_ids)):
            if idx + word_len >= len(input_ids):
                raise ValueError(f"Error: word: {word} is not in prompt!")
            
            if (input_ids[idx: idx+word_len] == word_idx).sum():
                word_ids.append(idx)
                break
    
    return word_ids, word_lens


def mask2bbox(mask):
    """
    Convert a binary mask into bounding boxes.
    
    Args:
        mask (Tensor): A tensor of shape [B, T, H, W], where B is the batch size, 
                       T is the time dimension, H is the height, and W is the width.
                       
    Returns:
        bbox (Tensor): A tensor of shape [B, T, 4] where each bounding box is represented as 
                        [top_left_y, top_left_x, bottom_right_y, bottom_right_x].
    """
    # Find the non-zero indices in the mask
    non_zero = mask.view(mask.shape[0], mask.shape[1], -1).nonzero()

    # Initialize bbox tensor
    bbox = torch.zeros((mask.shape[0], mask.shape[1], 4), dtype=torch.long)

    for b in range(mask.shape[0]):  # iterate over batch
        for t in range(mask.shape[1]):  # iterate over time steps
            # Find the min/max coordinates in the non-zero regions
            mask_t = mask[b, t]
            y_non_zero, x_non_zero = torch.where(mask_t != 0)
            
            if len(y_non_zero) > 0 and len(x_non_zero) > 0:  # check if there are any non-zero values
                top_left_y, top_left_x = y_non_zero.min(), x_non_zero.min()
                bottom_right_y, bottom_right_x = y_non_zero.max(), x_non_zero.max()
                
                # Store bounding box coordinates
                bbox[b, t] = torch.tensor([top_left_y.item(), top_left_x.item(),
                                           bottom_right_y.item(), bottom_right_x.item()])
    
    return bbox