#!/usr/bin/env python3
# Read through wsj files and construct vectors of (word-indices, tags)
# 

import argparse

def make_index( items ):
  items = list(items)
  items.sort()

  index = {}
  for i in range( len( items ) ):
    index[items[i]] = i

  return items, index

def parse( inputData, outputData, outputMap, limit = -1 ):
  words = set([])
  tags = set([])

  # Pass 1: Collect all words and tags
  idx = 0
  for l in open(inputData, 'r').readlines():
    # Split line. Each word is 'token_pos'
    (wordseq, tagseq) = zip(* map(lambda w: w.strip().split('_'), l.split()) )

    # Insert all words and tags into dictionary
    words.update( wordseq )
    tags.update( tagseq )

    idx += 1
    if( idx % 1000 == 0 ):
        print( "Read %d lines"%idx )
    if idx == limit: break

  print("Found %d words, %d tags"%( len(words), len(tags) ) )

  # Make index
  words, wordsIndex = make_index( words )
  tags, tagsIndex = make_index( tags )
  with open(outputMap + ".tags", 'w' ) as out:
    out.write( "\n".join(tags) )
    out.write( "\n" )
  with open(outputMap + ".words", 'w' ) as out:
    out.write( "\n".join(words) )
    out.write( "\n" )

  # Pass 2: Rewrite words in terms of these indices
  idx = 0
  with open(outputData + ".tags", 'w') as outTags:
    with open(outputData + ".words", 'w') as outWords:
      for l in open(inputData, 'r').readlines():
        # Split line. Each word is 'token_pos'
        (wordseq, tagseq) = zip(* map(lambda w: w.strip().split('_'), l.split()) )

        ts = map( lambda t: str( tagsIndex[t] ), tagseq ) 
        outTags.write( " ".join(ts) + "\n" )

        ws = map( lambda w: str( wordsIndex[w] ), wordseq ) 
        outWords.write( " ".join(ws) + "\n" )

        idx += 1 
        if( idx % 1000 == 0 ):
            print( "Read %d lines"%idx )
        if idx == limit: break
  print("Done.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Read WSJ files.')
  parser.add_argument('--input', metavar='input', type=str, required=True,
      help='input data file')
  parser.add_argument('--output', metavar='output', type=str, required=True,
      help='output data file')
  parser.add_argument('--index', metavar='index', type=str, required=True,
      help='output index file')
  parser.add_argument('--limit', metavar='limit', type=int, required=False, default=-1,
      help='limit the number of integers')
  args = parser.parse_args()

  parse( args.input, args.output, args.index, args.limit )

