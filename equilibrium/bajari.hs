module Bajari (
  lowerExt, upperExt, upperBoundBidsFunc
) where

import Numeric.Container
import Data.Random.Distribution.Uniform

-- Lower extremities
lowerExt ::
  Double      -- price weight (w)
  -> [Double] -- list of reputations
  -> [Double] -- corresponding list of lower extremities
lowerExt w = map (\r -> (1-w)*r)

-- Upper extremities
upperExt ::
  Double      -- price weight (w)
  -> [Double] -- list of reputations
  -> [Double] -- corresponding list of upper extremities
upperExt w reps = map (+w) $ lowerExt w reps

-- Upper bound on bids
upperBoundBidsFunc ::
  [Double]    -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> Double   -- output estimate on upper bound on bids
upperBoundBidsFunc lowers uppers =
  let bs = linspace 10000 (head uppers, uppers !! 1)
      cdfs = zipWith realUniformCDF (drop 1 lowers) (drop 1 uppers)
      negCdfs x = map (\cdf -> 1 - cdf x) cdfs
      objective x = (x - head uppers) * product (negCdfs x)
  in atIndex bs $ maxIndex $ mapVector objective bs
