import qualified Numeric.Container as NC
import qualified Numeric.GSL.Minimization as GSL
import qualified Test.HUnit as HUNIT
import qualified Data.String.Utils as UTILS
import qualified Foreign.Storable as FS
import qualified Bajari as B

-- Split list into list of sublists
split :: (Num a, FS.Storable a)
  => Int   -- length of a sublist
  -> Int   -- desired number of sublist
  -> [a]   -- input list
  -> [[a]] -- output list of sublists
split l n xs =
  let vXs = NC.fromList xs
      indexes = [0,l..(l*(n-1))]
  in map (NC.toList . (\i -> NC.subVector i l vXs)) indexes

-- (Scalar) cost function
costFunc ::
  Double               -- lower extremity
  -> Double            -- lower bound on bids
  -> NC.Vector Double  -- vector of coefficients
  -> Double            -- bid value
  -> Double            -- corresponding cost value
costFunc l bLow cs b =
  let k = NC.dim cs
      bs = NC.fromList $ zipWith (^) (take k [(b-bLow),(b-bLow)..]) [1..k]
  in l + NC.dot cs bs

-- Derivative of cost function
derivCostFunc ::
  Double               -- lower bound on bids
  -> NC.Vector Double  -- vector of coefficients
  -> Double            -- bids value
  -> Double            -- corresponding cost value
derivCostFunc bLow cs b =
  let k = NC.dim cs
      ps = zipWith (^) (take k [(b-bLow),(b-bLow)..]) [0..(k-1)]
      bs = NC.fromList $ zipWith (\x y -> x * fromIntegral y) ps [1..k]
  in NC.dot cs bs

-- FoC vector function
focFunc ::
  [Double]              -- list of lower extremities
  -> [Double]           -- list of upper extremities
  -> Double             -- lower bound on bids
  -> [NC.Vector Double] -- list of vector of coefficients
  -> Double             -- bid value
  -> NC.Vector Double   -- output FoC vector (to be minimized)
focFunc lowers uppers bLow vCs b =
  let n = length lowers
      costs = map (\(l,x) -> costFunc l bLow x b) $ zip lowers vCs
      derivCosts = NC.fromList $ map (\x -> derivCostFunc bLow x b) vCs
      probs = NC.fromList $ zipWith (-) uppers costs
      rs = NC.fromList $ map (\x -> 1 / (b - x)) costs
      consts = NC.constant (sum (NC.toList rs) / (fromIntegral n - 1)) n
  in NC.sub derivCosts $ NC.mul probs $ NC.sub consts rs

-- Upper boundary condition vector function
upperBoundFunc ::
  [Double]              -- list of lower extremities
  -> Double             -- lower bound on bids
  -> Double             -- upper bound on bids
  -> [NC.Vector Double] -- list of vector of coefficients
  -> NC.Vector Double   -- output upper boundary vector (to be minimized)
upperBoundFunc lowers bLow bUpper vCs =
  let n = length vCs
      costs = NC.fromList $ map (\(l,x) -> costFunc l bLow x bUpper) $ zip lowers vCs
      consts = NC.constant bUpper n
  in NC.sub costs consts

-- Objective function
objFunc ::
  Int         -- grid granularity
  -> Double   -- upper bound on bids
  -> [Double] -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> [Double] -- parameters to estimate
  -> Double   -- value of the objective
objFunc granularity bUpper lowers uppers params =
  let bLow = head params
      cs = drop 1 params
      n = length lowers
      m = length cs `div` fromIntegral n
      vCs = map NC.fromList $ split m n cs
      bs = NC.linspace granularity (bLow, bUpper)
      focSq b = NC.sumElements $ NC.mapVector (**2) $ focFunc lowers uppers bLow vCs b
      foc = NC.sumElements $ NC.mapVector focSq bs
      upperBound = NC.sumElements $ NC.mapVector (**2) $ upperBoundFunc lowers bLow bUpper vCs
  in foc + fromIntegral granularity * upperBound

{-
  Impure (main) program goes here
-}
-- Minimization
minimizeObj ::
  Int                     -- number of bidders
  -> Int                  -- current number of polynomial coefficients per bidder
  -> Int                  -- desired number of polynomial coefficients per bidder
  -> ([Double] -> Double) -- objective function
  -> [Double]             -- initial values for the parameters to estimate
  -> [Double]             -- initial size of the search box
  -> IO [Double]          -- minimized parameters
minimizeObj n i j objective params sizeBox = do
  let (s,_) = GSL.minimize GSL.NMSimplex2 1E-8 100000 sizeBox objective params
  print s
  if i == j
    then return s
    else do
      let b = head s
      let i' = i+1
      let sizeBox' = take (n*i' + 1) [1E-2,1E-2..]
      let cs' = split i n $ drop 1 s
      let params' = b : concatMap (++ [1E-6]) cs'
      minimizeObj n i' j objective params' sizeBox'

-- Main
main :: IO ()
main = do
  let w = 0.75
  let reps = [0.25, 0.5, 0.75]
  let n = length reps
  let numCoeffs = 3
  let desiredNumCoeffs = 8
  let lowers = B.lowerExt w reps
  let uppers = B.upperExt w reps
  let bUpper = B.upperBoundBidsFunc lowers uppers
  let granularity = 100
  let objective = objFunc granularity bUpper lowers uppers
  let l1 = (lowers !! 1) + 1E-3
  let initSizeBox = take (n*numCoeffs + 1) [1E-1,1E-1..]
  let initConditions = take (n*numCoeffs + 1) (l1 : [1E-2,1E-2..])
  s <- minimizeObj n numCoeffs desiredNumCoeffs objective initConditions initSizeBox
  let bLow = head s
  let cs = split desiredNumCoeffs n $ drop 1 s
  let filePath = "polynomial.out"
  let fileContents = UTILS.join "\n" [
        UTILS.join " " (["w", "reps", "b_lower", "b_upper"] ++ [UTILS.join "_" ["cs", show i] | i <- [0..n-1]]),
        UTILS.join " " ([show w, show reps, show bLow, show bUpper] ++ [show c | c <- cs])]
  writeFile filePath fileContents

{-
  Specification of tests goes here
-}
-- Test costFunc
testCostFunc :: HUNIT.Test
testCostFunc = HUNIT.TestCase (do
  let err = 1E-8
  let xs = [0.0,0.1..1.0]
  let expYs = map (\x -> x**2 - 0.5*x + 0.75) xs
  let ys = map (costFunc 0.5 0.5 (NC.fromList [0.25,0.5,1.0])) xs
  let cmp = all (== True) $ zipWith (\x y -> abs (x-y) < err) expYs ys
  HUNIT.assertBool "Testing costFunc: " cmp)
-- Test derivCostFunc
testDerivCostFunc :: HUNIT.Test
testDerivCostFunc = HUNIT.TestCase (do
  let err = 1E-8
  let xs = [0.0,0.1..1.0]
  let expYs = map (\x -> 2*x - 0.5) xs
  let ys = map (derivCostFunc 0.5 (NC.fromList [0.25,0.5,1.0])) xs
  let cmp = all (== True) $ zipWith (\x y -> abs (x-y) < err) expYs ys
  HUNIT.assertBool "Testing derivCostFunc: " cmp)

tests :: HUNIT.Test
tests = HUNIT.TestList [HUNIT.TestLabel "testCostFunc" testCostFunc,
                        HUNIT.TestLabel "testDerivCostFunc" testDerivCostFunc]
