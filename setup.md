```
cd ~/code/humanoid/ManipTrans/
module load conda/latest
conda activate maniptrans

unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP                                                                                                                                                                                                                             
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT                                                                                                                                                                                                                        
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF                                                                                                                                                                                                                        
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB                                                                                                                                                                                                                                  
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS                                                                                                                                                                                                                           
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
```