import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前 Git 仓库的最新提交（commit）的哈希值的前7个字符，并返回这个字符串。
# 如果当前目录不是 Git 仓库的一部分（即不存在 ../.git 目录），则函数返回 '0000000' 作为默认值
def get_git_commit_number():
    # 如果没找到，返回
    if not os.path.exists('../.git'):
        return '0000000'

    # 获取最新提交的哈希值并返回前7个字符
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number

# 创建一个CUDA扩展对象，用于在Python项目中编译和链接CUDA代码。
# 它接收三个参数：name（扩展的名称），module（模块名，用于确定源文件的路径），和sources（CUDA源文件列表）。
# 这个函数使用了CUDAExtension类（假设是从某个库如setuptools_cuda或torch.utils.cpp_extension中导入的），
# 来创建并返回一个配置好的CUDA扩展对象。
def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        # 构建源文件完整路径
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

# 将给定的版本号（version）写入到指定的目标文件（target_file）中。
# 函数使用了Python的with语句来打开文件，这确保了文件在写入后会被正确地关闭
# version: 一个字符串，表示要写入文件的版本号。
# target_file: 一个字符串，表示目标文件的路径，即版本号将要被写入哪个文件。
def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'al3d_utils/version.py')

    setup(
        name='al3d_utils',
        version=version,
        description='The global utils of the 3DAL pipeline',
        install_requires=[
            'numpy',
            'torch>=1.1',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='PJLAB-ADG',
        license='Apache License 2.0',
        packages=["al3d_utils"],
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',  # CUDA扩展名
                module='al3d_utils.ops.iou3d_nms',  # 源文件路径
                sources=[  # CUDA源文件列表
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='al3d_utils.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='al3d_utils.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='al3d_utils.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/ball_query_count.cpp',
                    'src/ball_query_count_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp',
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='al3d_utils.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='deform_conv_cuda',
                module='al3d_utils.ops.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu',

                ],
            ),
        ],
    )
