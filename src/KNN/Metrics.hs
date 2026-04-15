{-# OPTIONS_GHC -Wall #-}

-- | Метрики качества для оценки модели k-NN на размеченных данных.
module KNN.Metrics
  ( accuracy
  , meanSquaredError
  , formatReport
  ) where

import Data.Maybe (mapMaybe)

import KNN.Types (Label, PredictionResult (..))

-- | Точность классификации: доля правильно предсказанных меток.
-- Объекты без истинной метки игнорируются.
accuracy :: [PredictionResult] -> Double
accuracy results =
  case labelled of
    [] -> 0.0
    _  ->
      fromIntegral correct / fromIntegral (length labelled)
  where
    labelled = mapMaybe extractPair results
    correct  = length . filter (uncurry (==)) $ labelled

-- | Среднеквадратическая ошибка для задач регрессии.
-- Объекты без истинной метки игнорируются.
meanSquaredError :: [PredictionResult] -> Double
meanSquaredError results =
  case labelled of
    [] -> 0.0
    _  ->
      sum (map squaredErr labelled) / fromIntegral (length labelled)
  where
    labelled   = mapMaybe extractPair results
    squaredErr (predicted, actual) =
      (predicted - actual) ^ (2 :: Int)

-- | Сформировать читаемый отчёт об оценке качества модели.
formatReport :: [PredictionResult] -> String
formatReport results =
  unlines
    [ "=== Отчёт об оценке качества ==="
    , "Размеченных объектов : " ++ show numLabelled
    , "Accuracy             : " ++ show (accuracy results)
    , "MSE                  : " ++ show (meanSquaredError results)
    ]
  where
    numLabelled =
      length . filter (not . null . prActual) $ results

-- | Извлечь пару (предсказанная, истинная) метки, если истинная известна.
extractPair :: PredictionResult -> Maybe (Label, Label)
extractPair pr =
  case prActual pr of
    Nothing -> Nothing
    Just a  -> Just (prPredicted pr, a)
