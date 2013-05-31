#!/usr/bin/env python3
# Extract results from a .exec folder

import argparse
import os

# Set to the fields you're interested in.
OPTION_FIELDS = ( 
    'SpectralExperts.subsampleN',
    'SpectralExperts.inputPath',
    )
OUTPUT_FIELDS = ( 
    'PairsErr', 
    'TriplesErr', 
    'betasErr', 
    'betasEMErr', 
    )
LOG_FIELDS = ( 
    'Trace Reg:', 
    )

def read_fields( fname, fields ):
    data = {}

    with open( fname, 'r' ) as f:
        for line in f.readlines():
            line = line.strip()
            for k in fields:
                if line.startswith( k ):
                    [_,v] = line.split(k,1)
                    data[k] = v
    return data

def extract( folder ):
    # Read all the fields from the folder/options.map and
    # folder/output.map
    options_data = read_fields( os.path.join(folder, 'options.map'), OPTION_FIELDS )
    log_data = read_fields( os.path.join(folder, 'log'), LOG_FIELDS )
    output_data = read_fields( os.path.join(folder, 'output.map'), OUTPUT_FIELDS )
    return dict( list( options_data.items() ) + list( log_data.items() ) + list( output_data.items() ) + [('execDir', folder)] ) 

def printTable( data, keys=None ):
    """Print a dictionary of data as a tab-separated table, with column names as a comment"""
    if keys is None:
        keys = list( data.keys() )
    print("#", *keys, sep='\t')
    rows = len( data[keys[0]] )

    for i in range(rows):
        print( *[ data[k][i] for k in keys ], sep='\t' )

def main( ):
    parser = argparse.ArgumentParser( description='Extract fields from a .exec folder' )
    parser.add_argument( 'folder', type=str, nargs='+', help="Fields to extract from the folder" )

    args = parser.parse_args()
    FIELDS = (OPTION_FIELDS + LOG_FIELDS + OUTPUT_FIELDS + ('execDir',))
    data = dict( [ (k, []) for k in FIELDS ] )
    for folder in args.folder:
        data_ = extract(folder)
        for k in FIELDS:
            data[k].append( data_.get(k, '') )

    printTable(data, FIELDS)

if __name__ == "__main__":
    main()

