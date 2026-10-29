"""Convert trusted RRB demos into RGB Zarr without moving source files."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'responsible_robot_bench/envs/rrb_manipulation'),str(ROOT/'policies/PointFlowMatch')]
from pfp.data_utils.conversion import convert_to_zarr_direct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root',required=True,help='Embedded demo or upstream episode directory')
    parser.add_argument('--task_name',required=True)
    parser.add_argument('--output_path',required=True)
    parser.add_argument('--split',choices=['train','val'],default='train')
    parser.add_argument('--image_size',type=int,nargs=2,default=[128,128])
    args=parser.parse_args()
    result=convert_to_zarr_direct(args.data_root,args.task_name,args.output_path,tuple(args.image_size),args.split)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
