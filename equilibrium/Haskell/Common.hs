module Common (
  lowerExtremities,
  upperExtremities,
  upperBoundOnBids
) where

import Numeric.Container (atIndex, linspace, maxIndex, mapVector)
import Data.Random.Distribution.Uniform (realUniformCDF)

{-|
  The 'lowerExtremities' function computes lower extremities of the
  bidders (network operators). See Indirect Analysis of the network
  selection mechanism in DMP.
-}
lowerExtremities ::
  Double      -- price weight (w)
  -> [Double] -- list of reputations
  -> [Double] -- corresponding list of lower extremities
lowerExtremities w = map (\r -> (1-w)*r)

{-|
  The 'upperExtremities' function computes upper extremities of the
  bidders (network operators). See Indirect Analysis of the network
  selection mechanism in DMP.
-}
upperExtremities ::
  Double      -- price weight (w)
  -> [Double] -- list of reputations
  -> [Double] -- corresponding list of upper extremities
upperExtremities w = map (+w) . lowerExtremities w

{-|
  The 'upperBoundOnBids' function computes the upper bound on bids
  in the DMP auction. See Indirect Analysis of the network selection
  mechanism in DMP.
-}
upperBoundOnBids ::
  [Double]    -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> Double   -- output estimate on upper bound on bids
upperBoundOnBids lowers uppers =
  let bs = linspace 10000 (head uppers, uppers !! 1)
      cdfs = zipWith realUniformCDF (drop 1 lowers) (drop 1 uppers)
      negCdfs x = map (\cdf -> 1 - cdf x) cdfs
      objective x = (x - head uppers) * product (negCdfs x)
  in atIndex bs $ maxIndex $ mapVector objective bs
