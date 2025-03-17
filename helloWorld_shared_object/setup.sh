#!/bin/bash

###########################################################################################################
# Setup environments
###########################################################################################################
ARCH=$(uname -m) # x86_64
if [ -z ${CMSSW_SEARCH_PATH+x} ]; then # if CMSSW_SEARCH_PATH is unset
  if [ -z ${FORCED_CMSSW_VERSION+x} ]; then # if FORCED_CMSSW_VERSION is unset
    #export CMSSW_VERSION=CMSSW_14_2_0_pre4
    export CMSSW_VERSION=CMSSW_15_0_0_pre3 # change to current working path
  else
    export CMSSW_VERSION=$FORCED_CMSSW_VERSION
  fi

  source /cvmfs/cms.cern.ch/cmsset_default.sh 
  CMSSW_PATH=$(scram list -c CMSSW | grep -w $CMSSW_VERSION | awk '{print $3}')
  cd $CMSSW_PATH #/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_15_0_0_pre3
  echo "CMSSW_PATH: ${CMSSW_PATH}"
  eval `scramv1 runtime -sh` 
else
  cd $CMSSW_BASE/src
fi

# Export paths to libraries we need
export ALPAKA_ROOT=$(scram tool info alpaka | grep ALPAKA_BASE | cut -d'=' -f2)
export BOOST_ROOT=$(scram tool info boost | grep BOOST_BASE | cut -d'=' -f2)
export CUDA_HOME=$(scram tool info cuda | grep CUDA_BASE | cut -d'=' -f2)
export FMT_ROOT=$(scram tool info fmt | grep FMT_BASE | cut -d'=' -f2)
export ROCM_ROOT=$(scram tool info rocm | grep ROCM_BASE | cut -d'=' -f2)
export ROOT_ROOT=$(scram tool info root_interface | grep ROOT_INTERFACE_BASE | cut -d'=' -f2)
echo "ALPAKA_ROOT=${ALPAKA_ROOT}" # ALPAKA_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/alpaka/1.2.0-b081818336b70095080b83065d50ff0d
echo "BOOST_ROOT=${BOOST_ROOT}" # BOOST_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-96a02191111b66819e07de179cb25a0e
echo "CUDA_HOME=${CUDA_HOME}" # CUDA_HOME=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.8.0-15bfa86985d46d842bb5ecc3aca6c676
echo "FMT_ROOT=${FMT_ROOT}" # FMT_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/fmt/10.2.1-e35fd1db5eb3abc8ac0452e8ee427196
echo "ROCM_ROOT=${ROCM_ROOT}" # ROCM_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/rocm/6.3.2-94b981ba216f4b76c08c130cf3731d10
echo "ROOT_ROOT=${ROOT_ROOT}" # ROOT_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/lcg/root/6.32.09-0a945d6e24dcaabe218b38fa8292785d


cd - > /dev/null

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # back to LSTCore/standalone
export LD_LIBRARY_PATH=$DIR/LST:$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
#export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/ # YY: this does not exist
export TRACKINGNTUPLEDIR=/depot/cms/users/yao317/datasets/
export LSTOUTPUTDIR=.
