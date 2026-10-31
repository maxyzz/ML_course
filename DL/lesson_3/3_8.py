

def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    return (in_channels * kernel_size * kernel_size + (1 if bias else 0)) * out_channels


# print(count_parameters_conv(1024,6,3,True))
# print(count_parameters_conv(2048,2,8,False))