import Control.Applicative
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified Data.List as DL
import qualified Data.Maybe as DM
import qualified Data.String.Utils as UTILS
import qualified Bajari as B

-- Function for estimating k and c(b)
estimateKC ::
  Double
  -> NC.Vector Double
  -> (Int, Double)
estimateKC bLow lowers =
  let n = NC.dim lowers
      ks = [2..n]
      subListLowers = map (\i -> NC.subVector 0 i lowers) ks
      estimateC ls = bLow  - fromIntegral (NC.dim ls - 1) / NC.foldVector (\x acc -> acc + 1 / (bLow - x)) 0 ls
      costs = map estimateC subListLowers
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
  ((Double -> NC.Vector Double -> NC.Vector Double)
    -> NC.Vector Double
    -> NC.Vector Double
    -> NC.Matrix Double)
  -> Int
  -> NC.Vector Double
  -> NC.Vector Double
  -> NC.Vector Double
  -> NC.Matrix Double
  -> NC.Matrix Double
solveODE odeSolver k lowers uppers ts sol
  | k == NC.dim lowers = sol
  | otherwise = sol'
  where ext = extensionFunc k ts sol
        differences = NC.mapVector (\x -> abs (x - NC.atIndex lowers k)) ext
        stopIndex = NC.minIndex differences
        tempSol = map (NC.subVector 0 stopIndex) $ NC.toColumns sol ++ [ext]
        ts' = NC.subVector stopIndex (NC.dim ts - stopIndex) ts
        xdot' = focFunc (NC.subVector 0 (k+1) uppers)
        initials = NC.fromList $ map (`NC.atIndex` (stopIndex-1)) tempSol
        ode' = odeSolver xdot' initials ts'
        sol' = NC.fromColumns $ zipWith (\x y -> NC.join [x,y]) tempSol $ NC.toColumns ode'

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
        tss = ts guess
        (k, cost) = estimateKC (NC.atIndex tss 0) lowers
        initials = NC.mapVector (min cost) lowers
        xdot = focFunc (NC.subVector 0 k uppers)
        step = 0.01 * (tss NC.@> 1 - tss NC.@> 0)
        odeSolver = ODE.odeSolveV ODE.RKf45 step 1.49012E-6 1.49012E-6
        s = solveODE odeSolver k lowers uppers tss $ odeSolver xdot (NC.subVector 0 k initials) tss
        bids = NC.toList tss
        costs = map NC.toList $ NC.toColumns s
        inits = map head costs
        condition1 = concat $ zipWith (\l c -> map (\x -> l <= x && x <= bUpper) c) inits costs
        condition2 = concatMap (zipWith (>) bids) costs
        condition3 = zipWith (<) bids $ drop 1 bids

-- Main
main :: IO ()
main = do
  let w = 0.5
  let reps = [0.25, 0.5, 0.75]
  let n = length reps
  let lowers = B.lowerExt w reps
  let uppers = B.upperExt w reps
  let bUpper = B.upperBoundBidsFunc lowers uppers
  let ts low = NC.linspace 10000 (low, bUpper-0.1)
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
