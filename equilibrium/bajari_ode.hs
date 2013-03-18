import Control.Applicative
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified Data.String.Utils as UTILS
import qualified Bajari as B

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

-- Forward shooting method
forwardShooting ::
  Double                                              -- upper bound on bids
  -> (Double -> NC.Vector Double -> NC.Matrix Double) -- ODE solver
  -> Double                                           -- desired error
  -> (Double -> NC.Vector Double)                     -- grid function
  -> Double                                           -- lower bound on estimate
  -> Double                                           -- upper bound on estimate
  -> IO (Double, NC.Matrix Double)                    -- tuple of estimate and matrix of solutions
forwardShooting bUpper odeSolver err ts low high = do
  let guess = 0.5 * (low + high)
  let tss = ts guess
  let step = 0.01 * (tss NC.@> 1 - tss NC.@> 0)
  let s = odeSolver step tss
  if high - low < err
    then return (guess, s)
    else do
      let bids = NC.toList $ ts guess
      let costs = map NC.toList $ NC.toColumns s
      let inits = map head costs
      let condition1 = concat $ zipWith (\l c -> map (\x -> l <= x && x <= bUpper) c) inits costs
      let condition2 = concatMap (zipWith (>) bids) costs
      let condition3 = zipWith (<) bids $ drop 1 bids
      if and (condition1 ++ condition2 ++ condition3)
        then forwardShooting bUpper odeSolver err ts low guess
        else forwardShooting bUpper odeSolver err ts guess high

-- Main
main :: IO ()
main = do
  let w = 0.85
  let reps = [0.2, 0.4, 0.6, 0.8]
  let n = length reps
  let lowers = B.lowerExt w reps
  let uppers = B.upperExt w reps
  let bUpper = B.upperBoundBidsFunc lowers uppers
  let ts low = NC.linspace 1000 (low, bUpper-1E-1)
  let xdot = focFunc (NC.fromList uppers)
  let odeSolver step = ODE.odeSolveV ODE.RKf45 step 1.49012E-6 1.49012E-6 xdot (NC.fromList lowers)
  let low = lowers !! 1
  let high = bUpper
  let err = 1E-6
  (bLow, s) <- forwardShooting bUpper odeSolver err ts low high
  let bids = NC.toList $ ts bLow
  let costs = map (show . NC.toList) $ NC.toColumns s
  let filePath = "ode.out"
  let labels = UTILS.join " " (["w", "reps", "bids"] ++ [UTILS.join "_" ["costs", show i] | i <- [0..n-1]])
  let values = UTILS.join " " ([show w, show reps, show bids] ++ costs)
  let fileContents = UTILS.join "\n" [labels, values]
  writeFile filePath fileContents
