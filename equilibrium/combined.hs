import qualified Common as C
import qualified Data.List.Split as DLS
import qualified Data.String.Utils as DSU
import qualified ForwardShooting as FS
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified PolynomialProjection as PP

main :: IO ()
main = do
  -- prepare the scenario
  let w = 0.45
  let reps = [0.2, 0.4, 0.6, 0.8]
  let n = length reps
  let lowers = C.lowerExtremities w reps
  let uppers = C.upperExtremities w reps
  let bUpper = C.upperBoundOnBids lowers uppers
  -- solve using the Forward Shooting Method
  let ts low = NC.linspace 10000 (low, bUpper-0.013)
  let low = lowers !! 1
  let high = bUpper
  let err = 1E-6
  let (bLow, fsmSol) = FS.solve bUpper (NC.fromList lowers) (NC.fromList uppers) ODE.RKf45 err ts low high
  -- get the index when k = n
  let bids = ts bLow
  let costs = NC.toColumns fsmSol
  let index = NC.minIndex $ NC.mapVector (abs . (last lowers-)) $ last costs
  let bLow' = NC.atIndex bids index
  let lowers' = map (`NC.atIndex` index) costs
  -- solve the system when k = n using the Polynomial Projection Method
  let numCoeffs = 3
  let desiredNumCoeffs = 5
  let initSizeBox = take (n*numCoeffs) [1E-1,1E-1..]
  let initConditions = take (n*numCoeffs) [1E-2,1E-2..]
  let ppSol = PP.solve' bLow' bUpper lowers' uppers numCoeffs desiredNumCoeffs 100 initConditions initSizeBox
  -- combine the two results
  let cs = DLS.chunksOf desiredNumCoeffs ppSol
  let costFuncs x = zipWith (\c l -> PP.costFunction l bLow' (NC.fromList c) x) cs lowers'
  let tCosts = NC.toLists $ NC.fromColumns $ map (NC.subVector 0 index) costs
  let bids' = NC.linspace (NC.dim bids - index) (bLow', bUpper)
  let tCosts' = map costFuncs $ NC.toList bids'
  let solution = map (show . NC.toList) $ NC.toColumns $ NC.fromLists $ tCosts ++ tCosts'
  let solutionBids = NC.toList (NC.subVector 0 index bids) ++ NC.toList bids'
  let filePath = "combined.out"
  let labels = DSU.join " " (["w", "reps", "bids"] ++ [DSU.join "_" ["costs", show i] | i <- [0..n-1]])
  let values = DSU.join " " ([show w, show reps, show solutionBids] ++ solution)
  let fileContents = DSU.join "\n" [labels, values]
  writeFile filePath fileContents
