import qualified Common as C
import qualified Data.List.Split as DLS
import qualified Data.String.Utils as DSU
import qualified PolynomialProjection as PP

main :: IO ()
main = do
  let w = 0.75
  let reps = [0.25, 0.5, 0.75]
  let n = length reps
  let numCoeffs = 3
  let desiredNumCoeffs = 8
  let lowers = C.lowerExtremities w reps
  let uppers = C.upperExtremities w reps
  let bUpper = C.upperBoundOnBids lowers uppers
  let l1 = (lowers !! 1) + 1E-3
  let initSizeBox = take (n*numCoeffs + 1) [1E-1,1E-1..]
  let initConditions = take (n*numCoeffs + 1) (l1 : [1E-2,1E-2..])
  let s = PP.solve bUpper lowers uppers numCoeffs desiredNumCoeffs 100 initConditions initSizeBox
  print s
  let bLow = head s
  let cs = DLS.chunksOf desiredNumCoeffs $ drop 1 s
  let filePath = "polynomial.out"
  let fileContents = DSU.join "\n" [
        DSU.join " " (["w", "reps", "b_lower", "b_upper"] ++ [DSU.join "_" ["cs", show i] | i <- [0..n-1]]),
        DSU.join " " ([show w, show reps, show bLow, show bUpper] ++ [show c | c <- cs])]
  writeFile filePath fileContents
