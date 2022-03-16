#!/bin/bash
current_dir=$PWD
hardware_path=$current_dir/$1
part=$2

# change to the chisel directory
cd $FPGACONVNET_CHISEL

# get the number of partitions
NUM_PARTITIONS=$( jq '.partition | length' $hardware_path )

# iterate over partitions
for i in $( seq 1 ${NUM_PARTITIONS}); do

    # get current partition index
    partition_index=$(( $i - 1))

    # generate verilog
    ./scripts/build_partition.sh $hardware_path $partition_index

    # add attributes
    python scripts/add_attributes.py impl/PartitionTop.v impl/PartitionTop_attr.v

    # get resources
    vivado -mode batch -source scripts/get_rsc_usage.tcl -tclargs PartitionTop_attr PartitionTop $part

done

# go back to directory
cd $current_dir


