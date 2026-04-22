{-# OPTIONS_GHC -Wall #-}

-- | Метрики качества для оценки модели k-NN на размеченных данных.
module KNN.Metrics
  ( accuracy
  , meanSquaredError
  , formatReport
  ) where

import KNN.Types (PredictionResult (..))

-- | Точность классификации: доля правильно предсказанных меток.
accuracy :: [PredictionResult] -> Double
accuracy [] = 0.0
accuracy results =
  fromIntegral correct / fromIntegral (length results)
  where
    correct = length . filter (\pr -> prPredicted pr == prActual pr) $ results

-- | Среднеквадратическая ошибка для задач регрессии.
meanSquaredError :: [PredictionResult] -> Double
meanSquaredError [] = 0.0
meanSquaredError results =
  sum (map squaredErr results) / fromIntegral (length results)
  where
    squaredErr pr = (prPredicted pr - prActual pr) ^ (2 :: Int)

-- | Сформировать читаемый отчёт об оценке качества модели.
formatReport :: [PredictionResult] -> String
formatReport results =
  unlines
    [ "=== Отчёт об оценке качества ==="
    , "Объектов  : " ++ show (length results)
    , "Accuracy  : " ++ show (accuracy results)
    , "MSE       : " ++ show (meanSquaredError results)
    ]
