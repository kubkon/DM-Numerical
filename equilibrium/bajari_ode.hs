import Control.Applicative
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified Data.List as DL
import qualified Data.Maybe as DM
import qualified Data.String.Utils as UTILS
import qualified Bajari as B

-- Function for estimating c(b)
estimateC ::
  Double -- vector of lower extremities
  -> NC.Vector Double        -- lower bound on bids
  -> Double        -- extended cost
estimateC bLow lowers =
  let n = NC.dim lowers
      sum' = NC.foldVector (\x acc -> acc + 1 / (bLow - x)) 0 lowers
  in bLow - (fromIntegral n - 1) / sum'

-- Function for estimating k and c(b)
estimateKC ::
  Double
  -> NC.Vector Double
  -> (Int, Double)
estimateKC bLow lowers =
  let n = NC.dim lowers
      ks = [2..n]
      subListLowers = map (\i -> NC.subVector 0 i lowers) ks
      costs = map (estimateC bLow) subListLowers
      candidates = zip ks costs
      test (k, cost)
        | k < n = (listLowers !! (k-1) <= cost) && (cost < listLowers !! k)
        | otherwise = listLowers !! (k-1) <= cost
        where listLowers = NC.toList lowers
      (result, _) = DM.fromJust $ DL.find snd $ zip candidates $ map test candidates
  in result

-- Extension vector function
extensionFunc ::
  Int
  -> NC.Vector Double
  -> NC.Matrix Double
  -> NC.Vector Double
extensionFunc k bids odeSol =
  let costsPerBid = NC.toRows odeSol
      sum' b = NC.foldVector (\c acc -> acc + 1 / (b - c)) 0
      sums = zipWith (\costs b -> b - (fromIntegral k - 1) / sum' b costs) costsPerBid $ NC.toList bids
  in NC.fromList sums

-- FoC vector function
focFunc :: 
  NC.Vector Double     -- vector of upper extremities
  -> Double            -- independent variable
  -> NC.Vector Double  -- vector of inputs
  -> NC.Vector Double  -- vector of derivatives
focFunc uppers t ys =
  let n = NC.dim ys
      probV = NC.zipVectorWith (-) uppers ys
      rsV = NC.mapVector (\x -> 1 / (t - x)) ys
      constV = NC.constant (sum (NC.toList rsV) / (fromIntegral n - 1)) n
  in NC.mul probV $ NC.sub constV rsV

-- Solve ODE function
solveODE ::
  NC.Vector Double
  -> NC.Vector Double
  -> NC.Vector Double
  -> NC.Matrix Double
solveODE lowers uppers ts
  | k < n = solution'
  | otherwise = ode
  where n = NC.dim lowers
        (k, cost) = estimateKC (NC.atIndex ts 0) lowers
        initials = NC.mapVector (min cost) lowers
        xdot = focFunc (NC.subVector 0 k uppers)
        step = 0.01 * (ts NC.@> 1 - ts NC.@> 0)
        ode = ODE.odeSolveV ODE.RKf45 step 1.49012E-6 1.49012E-6 xdot (NC.subVector 0 k initials) ts
        ext = extensionFunc k ts ode
        differences = NC.mapVector (\x -> abs (x - NC.atIndex lowers k)) ext
        stopIndex = NC.minIndex differences
        solution = map (NC.subVector 0 stopIndex) $ NC.toColumns ode ++ [ext]
        ts' = NC.subVector stopIndex (NC.dim ts - stopIndex) ts
        xdot' = focFunc uppers
        initials' = NC.fromList $ map (`NC.atIndex` (stopIndex-1)) solution
        ode' = ODE.odeSolveV ODE.RKf45 step 1.49012E-6 1.49012E-6 xdot' initials' ts'
        solution' = NC.fromColumns $ zipWith (\x y -> NC.join [x,y]) solution $ NC.toColumns ode'

-- Forward shooting method
forwardShooting ::
  Double                                              -- upper bound on bids
  -> NC.Vector Double                                 -- vector of lower extremities
  -> NC.Vector Double                                 -- vector of upper extremities
  -> Double                                           -- desired error
  -> (Double -> NC.Vector Double)                     -- grid function
  -> Double                                           -- lower bound on estimate
  -> Double                                           -- upper bound on estimate
  -> (Double, NC.Matrix Double)                       -- tuple of estimate and matrix of solutions
forwardShooting bUpper lowers uppers err ts low high
  | high - low < err = (guess, s)
  | and (condition1 ++ condition2 ++ condition3) = forwardShooting bUpper lowers uppers err ts low guess
  | otherwise = forwardShooting bUpper lowers uppers err ts guess high
  where guess = 0.5 * (low + high)
        s = solveODE lowers uppers $ ts guess
        bids = NC.toList $ ts guess
        costs = map NC.toList $ NC.toColumns s
        inits = map head costs
        condition1 = concat $ zipWith (\l c -> map (\x -> l <= x && x <= bUpper) c) inits costs
        condition2 = concatMap (zipWith (>) bids) costs
        condition3 = zipWith (<) bids $ drop 1 bids

-- Main
main :: IO ()
main = do
  let w = 0.75
  let reps = [0.25, 0.5, 0.75]
  let n = length reps
  let lowers = B.lowerExt w reps
  let uppers = B.upperExt w reps
  let bUpper = B.upperBoundBidsFunc lowers uppers
  let ts low = NC.linspace 10000 (low, bUpper-0.09)
  let low = lowers !! 1
  let high = bUpper
  let err = 1E-6
  let (bLow, s) = forwardShooting bUpper (NC.fromList lowers) (NC.fromList uppers) err ts low high
  let bids = NC.toList $ ts bLow
  let costs = map (show . NC.toList) $ NC.toColumns s
  let filePath = "ode.out"
  let labels = UTILS.join " " (["w", "reps", "bids"] ++ [UTILS.join "_" ["costs", show i] | i <- [0..n-1]])
  let values = UTILS.join " " ([show w, show reps, show bids] ++ costs)
  let fileContents = UTILS.join "\n" [labels, values]
  writeFile filePath fileContents
