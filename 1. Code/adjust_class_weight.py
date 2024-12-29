def calculate_scale_pos_weight(y):
    """Calculate scale_pos_weight based on class distribution"""
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    return neg_count / pos_count