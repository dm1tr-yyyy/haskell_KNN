{-# OPTIONS_GHC -Wall #-}

-- | Алгоритм k ближайших соседей: вычисление расстояний,
-- поиск соседей и предсказание метки.
module KNN.Algorithm
  ( trainModel
  , predict
  , predictAll
  , computeDistance
  , findKNearest
  ) where

import Data.List (sortBy)
import Data.Ord  (comparing)

import KNN.Types
  ( Config (..)
  , DataPoint (..)
  , DistanceMetric (..)
  , Features
  , Label
  , Model (..)
  , PredictionResult (..)
  , TaskType (..)
  , TrainingSet
  )

-- | Построить модель, сохранив обучающую выборку вместе с настройками.
trainModel :: Config -> TrainingSet -> Either String Model
trainModel cfg ts
  | configK cfg <= 0 =
      Left "Параметр k должен быть положительным целым числом"
  | null ts =
      Left "Обучающая выборка пуста"
  | configK cfg > length ts =
      Left "Параметр k превышает количество объектов в обучающей выборке"
  | otherwise =
      Right Model { modelConfig = cfg, modelTrainingSet = ts }

-- | Предсказать метку для одного вектора признаков с помощью модели.
-- Для классификации используется голосование большинством,
-- для регрессии — среднее значений соседей.
predict :: Model -> Features -> Either String Label
predict model features =
  case findKNearest model features of
    Left err        -> Left err
    Right neighbors ->
      let labels = map dpLabel neighbors
      in Right $ case configTaskType (modelConfig model) of
           Classification -> majorityVote labels
           Regression     -> meanValue labels

-- | Предсказать метки для списка объектов.
-- Объекты могут содержать истинную метку (для оценки качества).
predictAll
  :: Model
  -> [(Features, Maybe Label)]
  -> [Either String PredictionResult]
predictAll model = map predictOne
  where
    predictOne (fs, actual) =
      case predict model fs of
        Left err  -> Left err
        Right lbl -> Right PredictionResult
          { prFeatures  = fs
          , prPredicted = lbl
          , prActual    = actual
          }

-- | Найти k ближайших объектов обучающей выборки к запросу.
findKNearest :: Model -> Features -> Either String [DataPoint]
findKNearest model query
  | length query /= expectedDim =
      Left ( "Несовпадение размерности признаков: ожидалось "
             ++ show expectedDim
             ++ ", получено "
             ++ show (length query) )
  | otherwise =
      Right . take k . sortBy (comparing dist) $ trainingSet
  where
    cfg         = modelConfig model
    trainingSet = modelTrainingSet model
    k           = configK cfg
    metric      = configMetric cfg
    expectedDim = case trainingSet of
                    []      -> 0
                    (dp: _) -> length (dpFeatures dp)
    dist dp     = computeDistance metric query (dpFeatures dp)

-- | Вычислить расстояние между двумя векторами признаков по заданной метрике.
computeDistance :: DistanceMetric -> Features -> Features -> Double
computeDistance metric xs ys =
  case metric of
    Euclidean -> euclideanDist xs ys
    Manhattan -> manhattanDist xs ys
    Chebyshev -> chebyshevDist xs ys

-- | Евклидово расстояние (L2).
euclideanDist :: Features -> Features -> Double
euclideanDist xs ys =
  sqrt . sum $ zipWith (\x y -> (x - y) ^ (2 :: Int)) xs ys

-- | Манхэттенское расстояние (L1).
manhattanDist :: Features -> Features -> Double
manhattanDist xs ys =
  sum $ zipWith (\x y -> abs (x - y)) xs ys

-- | Расстояние Чебышёва (L-inf).
chebyshevDist :: Features -> Features -> Double
chebyshevDist xs ys =
  maximum $ zipWith (\x y -> abs (x - y)) xs ys

-- | Среднее значение меток соседей (для задач регрессии).
meanValue :: [Label] -> Label
meanValue [] = 0.0
meanValue ls = sum ls / fromIntegral (length ls)

-- | Классификация голосованием большинства среди соседей.
-- При равенстве голосов выбирается наименьшая метка (детерминированность).
majorityVote :: [Label] -> Label
majorityVote [] = 0.0
majorityVote ls =
  case sortBy cmp counts of
    []              -> 0.0
    ((winner, _):_) -> winner
  where
    unique  = foldr insertUniq [] ls
    counts  = map (\l -> (l, length (filter (== l) ls))) unique
    cmp (l1, c1) (l2, c2) =
      case compare c2 c1 of
        EQ -> compare l1 l2
        r  -> r
    insertUniq x acc = if x `elem` acc then acc else x : acc
