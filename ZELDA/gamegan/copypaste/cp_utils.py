import torch.nn.functional as F
import torch

def extract_interpolated_mask(mask_copy, target_size):
    """
    downsample de mask naar v dim
    interpolate bilinear
    expects mask of shape 1,1,h,w and v of shape 1,C,h,w
    """
    interpolated_mask = F.interpolate(mask_copy, size=target_size, mode='bilinear')
    return interpolated_mask

def get_bounds(mask):
    """
    get the bounds of the mask
    """
    y, x = mask.shape[2:]
    t, l, b, r = y+1, x+1, -1, -1
    
    for i in range(y):
        for j in range(x):
            val = mask[0, 0, i, j].item()
            if val > 0.0:
                t = min(i, t)
                b = max(i, b)
                l = min(j, l)
                r = max(j, r)
    
    return t, l, b, r

def center_bounds(t, l, b, r):
    """
    center the bounds of the mask around middle
    """
    y_mid = round((t + b) / 2)
    x_mid = round((l + r) / 2)
    
    return t-y_mid, l-x_mid, b-y_mid, r-x_mid

def get_center(mask):
    y, x = mask.shape[2:]
    
    y_count = 0
    x_count = 0
    total_count = 0
    
    for i in range(y):
        for j in range(x):
            val = mask[0, 0, i, j].item()
            if val > 0.0:
                y_count += i
                x_count += j
                total_count += 1
    
    if total_count == 0:
        raise ValueError

    return round(y_count/total_count), round(x_count/total_count)

def get_paste_bounds(interpolated_copy_mask, interpolated_paste_mask):
    t_old, l_old, b_old, r_old = get_bounds(interpolated_copy_mask)
    t_centered, l_centered, b_centered, r_centered = center_bounds(t_old, l_old, b_old, r_old)
    hmid, wmid = get_center(interpolated_paste_mask)
    t_new, l_new, b_new, r_new = t_centered+hmid, l_centered+wmid, b_centered+hmid, r_centered+wmid
    return t_new, l_new, b_new, r_new

def merge_old_with_new(new_v, old_v, mask, t_old, l_old, b_old, r_old, t_new, l_new, b_new, r_new):
    target_v = torch.clone(old_v)
    y_diff = t_new - t_old
    x_diff = l_new - l_old
    for i in range(t_new, b_new+1):
        if i < 0 or i >= target_v.shape[2]:
            continue
        for j in range(l_new, r_new+1):
            if j < 0 or j >= target_v.shape[3]:
                continue
            #target_v[0, :, i, j] = target_v[0, :, i, j]*(1 - paste_mask[0, 0, i-y_diff, j-x_diff]) + new_v[0, :, i-y_diff, j-x_diff]
            #target_v[0, :, i, j] = new_v[0, :, i-y_diff, j-x_diff]
            target_v[0, :, i, j] = target_v[0, :, i, j]*(1 - mask[0, 0, i-y_diff, j-x_diff]) + new_v[0, :, i-y_diff, j-x_diff]
    return target_v

def move_copy_v_to_paste_center(copy_mask, paste_mask, copy_v):
    interpolated_copy_mask = extract_interpolated_mask(copy_mask, copy_v.shape[2:])
    new_unmoved_v = torch.mul(copy_v, interpolated_copy_mask)

    interpolated_paste_mask = extract_interpolated_mask(paste_mask, copy_v.shape[2:])
    
    t_old, l_old, b_old, r_old = get_bounds(interpolated_copy_mask)
    t_new, l_new, b_new, r_new = get_paste_bounds(interpolated_copy_mask, interpolated_paste_mask)

    new_v = torch.zeros(new_unmoved_v.shape)
    y_diff = t_new - t_old
    x_diff = l_new - l_old
    for i in range(t_new, b_new+1):
        if i < 0 or i >= new_v.shape[2]:
            continue
        for j in range(l_new, r_new+1):
            if j < 0 or j >= new_v.shape[3]:
                continue
            new_v[0, :, i, j] = new_unmoved_v[0, :, i-y_diff, j-x_diff]
    return new_v

def get_v_from_selection(copy_mask, copy_v, paste_mask, paste_v):
    interpolated_copy_mask = extract_interpolated_mask(copy_mask, copy_v.shape[2:])
    new_v = torch.mul(copy_v, interpolated_copy_mask)

    interpolated_paste_mask = extract_interpolated_mask(paste_mask, paste_v.shape[2:])

    t_old, l_old, b_old, r_old = get_bounds(interpolated_copy_mask)
    t_new, l_new, b_new, r_new = get_paste_bounds(interpolated_copy_mask, interpolated_paste_mask)

    target_v = merge_old_with_new(new_v, paste_v, interpolated_copy_mask, t_old, l_old, b_old, r_old, t_new, l_new, b_new, r_new)
    return target_v

def test():
    #test_mask = torch.randint(0, 2, (1, 1, 16, 11), dtype=torch.float32)
    test_copy_mask = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 1, 0, 0,
                            0, 0, 0, 0, 1, 1, 1, 0,
                            0, 0, 0, 1, 1, 1, 0, 0,
                            0, 0, 0, 0, 1, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0])
    test_copy_mask = test_copy_mask.view(1,1,8,8)
    copy_v = torch.randn((1, 64, 5, 5))

    test_paste_mask = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 0,
                            0, 1, 1, 0, 0, 0, 0, 0,
                            0, 1, 1, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0])
    test_paste_mask = test_paste_mask.view(1,1,8,8)
    paste_v = torch.randn((1, 64, 5, 5))

    target_v = get_v_from_selection(test_copy_mask, copy_v, test_paste_mask, paste_v)
    print(target_v[0,0])

if __name__ == "__main__":
    test()