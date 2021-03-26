import torch
import cupy as cp
import cupy
from string import Template

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

scaleCUDA = '''
extern "C" __global__
void scaleBIFM(float *sIFM, float * bIFM, float * coef_sparse, unsigned *col_idx, unsigned *row_idx) {
  unsigned coef_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned w = blockIdx.z * blockDim.z + threadIdx.z;

  if (coef_idx < ${coef_len} && h < ${H} && w < ${W}) {
    unsigned hw_offset = (h*${W}) + w;
    unsigned sIFM_pix_offset = hw_offset * ${coef_len};
    unsigned b = col_idx[coef_idx];
    unsigned d = row_idx[coef_idx];

    // NOTE: bIFM is being pulled into cache repeatedly, so we may need to import it as __local__ for efficient memory access
    float *bIFM_targ = &bIFM[(b*${DHW}) + (d*${HW}) + hw_offset];
    //sIFM[sIFM_pix_offset + coef_idx] = *bIFM_targ * coef_sparse[coef_idx];
  }
}
'''#, 'scaleBIFM')

# accumulate kernel could be done with a reduction kernel IF dimensions are consistent
# in this case, sparse matrices make dimension reduction impossible
# maybe, we should consider having a cuboid sOFM matrix? WARNING -- this would allocate a HUGE amount of data in the GPU [~1 GB] for only ONE layer

# accumulateCUDA assumes a SPARSE kernel, and becomes inefficient the less sparse it is
accumulateCUDA = '''
extern "C" __global__
void accSIFM (float * OFM, float *sIFM, unsigned *acc_ptrs) {
  unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned h = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned w = blockIdx.z * blockDim.z + threadIdx.z;

  if (c < ${C} && h < ${H} && w < ${W}) {
    unsigned hw_offset = h*${W} + w;
    unsigned sIFM_pix_offset = hw_offset * ${coef_len};

    float * OFM_targ = &OFM[c*${HW} + hw_offset];
    for (unsigned i = acc_ptrs[c]; i < acc_ptrs[c+1]; i++) {
      *OFM_targ += sIFM[sIFM_pix_offset + i];
    }
  }
}
'''

