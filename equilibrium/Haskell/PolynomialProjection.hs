module PolynomialProjection (
  solve,
  solve',
  costFunction
) where

import Data.List.Split (chunksOf)
import qualified Numeric.Container as NC
import Numeric.GSL.Minimization (minimize,MinimizeMethod(NMSimplex2))
import qualified Test.HUnit as HUNIT

-- Minimization
solve ::
  Double      -- upper bound on bids
  -> [Double] -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> Int      -- current number of polynomial coefficients per bidder
  -> Int      -- desired number of polynomial coefficients per bidder
  -> Int      -- granularity
  -> [Double] -- initial values for the parameters to estimate
  -> [Double] -- initial size of the search box
  -> [Double] -- minimized parameters
solve bUpper lowers uppers i j granularity params sizeBox
  | i == j = s
  | otherwise = solve bUpper lowers uppers i' j granularity params' sizeBox'
  where (s,_) = minimize NMSimplex2 1E-8 100000 sizeBox obj params
        obj = objective granularity bUpper lowers uppers
        b = head s
        i' = i+1
        sizeBox' = take (length lowers * i' + 1) [1E-2,1E-2..]
        cs = chunksOf i $ drop 1 s
        params' = b : concatMap (++ [1E-6]) cs

solve' ::
  Double      -- lower bound on bids
  -> Double   -- upper bound on bids
  -> [Double] -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> Int      -- current number of polynomial coefficients per bidder
  -> Int      -- desired number of polynomial coefficients per bidder
  -> Int      -- granularity
  -> [Double] -- initial values for the parameters to estimate
  -> [Double] -- initial size of the search box
  -> [Double] -- minimized parameters
solve' bLow bUpper lowers uppers i j granularity params sizeBox
  | i == j = s
  | otherwise = solve' bLow bUpper lowers uppers i' j granularity params' sizeBox'
  where (s,_) = minimize NMSimplex2 1E-8 100000 sizeBox obj params
        obj = objective' granularity bLow bUpper lowers uppers
        i' = i+1
        sizeBox' = take (length lowers * i') [1E-2,1E-2..]
        params' = concatMap (++ [1E-6]) $ chunksOf i s

-- (Scalar) cost function
costFunction ::
  Double               -- lower extremity
  -> Double            -- lower bound on bids
  -> NC.Vector Double  -- vector of coefficients
  -> Double            -- bid value
  -> Double            -- corresponding cost value
costFunction l bLow cs b =
  let k = NC.dim cs
      bs = NC.fromList $ zipWith (^) (take k [(b-bLow),(b-bLow)..]) [1..k]
  in l + NC.dot cs bs

-- Derivative of cost function
derivativeCostFunction ::
  Double               -- lower bound on bids
  -> NC.Vector Double  -- vector of coefficients
  -> Double            -- bids value
  -> Double            -- corresponding cost value
derivativeCostFunction bLow cs b =
  let k = NC.dim cs
      ps = zipWith (^) (take k [(b-bLow),(b-bLow)..]) [0..(k-1)]
      bs = NC.fromList $ zipWith (\x y -> x * fromIntegral y) ps [1..k]
  in NC.dot cs bs

-- FoC vector function
firstOrderCondition ::
  [Double]              -- list of lower extremities
  -> [Double]           -- list of upper extremities
  -> Double             -- lower bound on bids
  -> [NC.Vector Double] -- list of vector of coefficients
  -> Double             -- bid value
  -> NC.Vector Double   -- output FoC vector (to be minimized)
firstOrderCondition lowers uppers bLow vCs b =
  let n = length lowers
      costs = map (\(l,x) -> costFunction l bLow x b) $ zip lowers vCs
      derivCosts = NC.fromList $ map (\x -> derivativeCostFunction bLow x b) vCs
      probs = NC.fromList $ zipWith (-) uppers costs
      rs = NC.fromList $ map (\x -> 1 / (b - x)) costs
      consts = NC.constant (sum (NC.toList rs) / (fromIntegral n - 1)) n
  in NC.sub derivCosts $ NC.mul probs $ NC.sub consts rs

-- Upper boundary condition vector function
upperBoundaryCondition ::
  [Double]              -- list of lower extremities
  -> Double             -- lower bound on bids
  -> Double             -- upper bound on bids
  -> [NC.Vector Double] -- list of vector of coefficients
  -> NC.Vector Double   -- output upper boundary vector (to be minimized)
upperBoundaryCondition lowers bLow bUpper vCs =
  let n = length vCs
      costs = NC.fromList $ map (\(l,x) -> costFunction l bLow x bUpper) $ zip lowers vCs
      consts = NC.constant bUpper n
  in NC.sub costs consts

-- Objective function
objective ::
  Int         -- grid granularity
  -> Double   -- upper bound on bids
  -> [Double] -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> [Double] -- parameters to estimate
  -> Double   -- value of the objective
objective granularity bUpper lowers uppers params =
  let bLow = head params
      cs = drop 1 params
      n = length lowers
      m = length cs `div` fromIntegral n
      vCs = map NC.fromList $ chunksOf m cs
      bs = NC.linspace granularity (bLow, bUpper)
      focSq b = NC.sumElements $ NC.mapVector (**2) $ firstOrderCondition lowers uppers bLow vCs b
      foc = NC.sumElements $ NC.mapVector focSq bs
      upperBound = NC.sumElements $ NC.mapVector (**2) $ upperBoundaryCondition lowers bLow bUpper vCs
  in foc + fromIntegral granularity * upperBound

objective' ::
  Int         -- grid granularity
  -> Double   -- lower bound on bids
  -> Double   -- upper bound on bids
  -> [Double] -- list of lower extremities
  -> [Double] -- list of upper extremities
  -> [Double] -- parameters to estimate
  -> Double   -- value of the objective
objective' granularity bLow bUpper lowers uppers params =
  let n = length lowers
      m = length params `div` fromIntegral n
      vParams = map NC.fromList $ chunksOf m params
      bs = NC.linspace granularity (bLow, bUpper)
      focSq b = NC.sumElements $ NC.mapVector (**2) $ firstOrderCondition lowers uppers bLow vParams b
      foc = NC.sumElements $ NC.mapVector focSq bs
      upperBound = NC.sumElements $ NC.mapVector (**2) $ upperBoundaryCondition lowers bLow bUpper vParams
  in foc + fromIntegral granularity * upperBound

-- Test costFunction
testCostFunction :: HUNIT.Test
testCostFunction = HUNIT.TestCase (do
  let err = 1E-8
  let xs = [0.0,0.1..1.0]
  let expYs = map (\x -> x**2 - 0.5*x + 0.75) xs
  let ys = map (costFunction 0.5 0.5 (NC.fromList [0.25,0.5,1.0])) xs
  let cmp = all (== True) $ zipWith (\x y -> abs (x-y) < err) expYs ys
  HUNIT.assertBool "Testing costFunction: " cmp)

-- Test derivCostFunc
testDerivativeCostFunction :: HUNIT.Test
testDerivativeCostFunction = HUNIT.TestCase (do
  let err = 1E-8
  let xs = [0.0,0.1..1.0]
  let expYs = map (\x -> 2*x - 0.5) xs
  let ys = map (derivativeCostFunction 0.5 (NC.fromList [0.25,0.5,1.0])) xs
  let cmp = all (== True) $ zipWith (\x y -> abs (x-y) < err) expYs ys
  HUNIT.assertBool "Testing derivativeCostFunction: " cmp)

tests :: HUNIT.Test
tests = HUNIT.TestList [HUNIT.TestLabel "testCostFunction" testCostFunction,
                        HUNIT.TestLabel "testDerivativeOfCostFunction" testDerivativeCostFunction]
