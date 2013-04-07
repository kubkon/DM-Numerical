import qualified Common as C
import qualified Control.Spoon as CS
import qualified Data.List.Split as DLS
import qualified Data.String.Utils as DSU
import qualified ForwardShooting as FS
import qualified Numeric.GSL.ODE as ODE
import qualified Numeric.Container as NC
import qualified PolynomialProjection as PP
import qualified System.Environment as SE

fsmHelper ::
  Double                                  -- upper bound on bids
  -> [Double]                             -- list of lower extremities
  -> [Double]                             -- list of upper extremities
  -> Double                               -- input Lipschitz parameter
  -> (Double, (Double, NC.Matrix Double)) -- output Lipshitz parameter, lower bound on bids and ODE solution matrix
fsmHelper bUpper lowers uppers param = case fsmOutput of
  Just (x,y) -> (param, (x,y))
  Nothing -> fsmHelper bUpper lowers uppers $ param + 0.001
  where fsmOutput = CS.teaspoon $
                    FS.solve bUpper (NC.fromList lowers) (NC.fromList uppers) ODE.RKf45 1E-6 ts (lowers !! 1) bUpper
        ts l = NC.linspace 10000 (l, bUpper - param)

main :: IO ()
main = do
  params <- SE.getArgs
  let params' = map read params :: [Double]
  let param = head params'
  let w = params' !! 1
  let reps = tail $ tail params'
  -- prepare the scenario
  let n = length reps
  let lowers = C.lowerExtremities w reps
  let uppers = C.upperExtremities w reps
  let bUpper = C.upperBoundOnBids lowers uppers
  -- solve using the Forward Shooting Method
  let (param', (bLow, fsmSol)) = fsmHelper bUpper lowers uppers param
  let ts low = NC.linspace 10000 (low, bUpper - param')
  -- get the index when k = n
  let bids = ts bLow
  let costs = NC.toColumns fsmSol
  let index = NC.minIndex $ NC.mapVector (abs . (last lowers-)) $ last costs
  let bLow' = NC.atIndex bids index
  let lowers' = map (`NC.atIndex` index) costs
  -- solve the system when k = n using the Polynomial Projection Method
  let numCoeffs = 3
  let desiredNumCoeffs = 8
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
