import torch

# Define all possible classes as a module-level constant
LAND_USE_CLASSES = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

def one_hot_encode_land_use(class_list):
    """
    Convert a list of land use class names into a single one-hot encoded vector using PyTorch.
    Each class in the input list gets marked with a 1 in the output tensor.
    
    Args:
        class_list (list): List of land use class names
        
    Returns:
        torch.Tensor: 1D tensor where 1 indicates presence of a class
    """
    # Create a dictionary mapping class names to indices
    class_to_idx = {class_name: idx for idx, class_name in enumerate(LAND_USE_CLASSES)}
    
    # Initialize the output tensor
    output = torch.zeros(len(LAND_USE_CLASSES), dtype=torch.long)
    
    # Set 1s for each class in the input list
    for class_name in class_list:
        if class_name in class_to_idx:
            output[class_to_idx[class_name]] = 1
        else:
            raise ValueError(f"Unknown class name: {class_name}")
    
    return output

def decode_land_use(one_hot_tensor):
    """
    Convert one-hot encoded tensor back to list of class names.
    
    Args:
        one_hot_tensor (torch.Tensor): One-hot encoded 1D tensor
        
    Returns:
        list: List of class names where the tensor had 1s
    """
    # Make sure input is a tensor
    if not isinstance(one_hot_tensor, torch.Tensor):
        one_hot_tensor = torch.tensor(one_hot_tensor)
    
    # Get the indices where the tensor is 1
    indices = torch.where(one_hot_tensor == 1)[0]
    
    # Convert indices to class names
    return [LAND_USE_CLASSES[idx] for idx in indices]