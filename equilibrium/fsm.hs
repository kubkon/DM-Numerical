import qualified Common as C
import qualified ForwardShooting as FS
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified Data.String.Utils as DSU

main :: IO ()
main = do
  let w = 0.45
  let reps = [0.2, 0.4, 0.6, 0.8]
  let n = length reps
  let lowers = C.lowerExtremities w reps
  let uppers = C.upperExtremities w reps
  let bUpper = C.upperBoundOnBids lowers uppers
  let ts low = NC.linspace 10000 (low, bUpper-0.013)
  let low = lowers !! 1
  let high = bUpper
  let err = 1E-6
  let (bLow, s) = FS.solve bUpper (NC.fromList lowers) (NC.fromList uppers) ODE.RKf45 err ts low high
  let bids = NC.toList $ ts bLow
  let costs = map (show . NC.toList) $ NC.toColumns s
  let filePath = "ode.out"
  let labels = DSU.join " " (["w", "reps", "bids"] ++ [DSU.join "_" ["costs", show i] | i <- [0..n-1]])
  let values = DSU.join " " ([show w, show reps, show bids] ++ costs)
  let fileContents = DSU.join "\n" [labels, values]
  writeFile filePath fileContents
