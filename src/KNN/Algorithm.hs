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
predict :: Model -> Features -> Label
predict model features =
  let pairs = map (\(dp, d) -> (dpLabel dp, d)) (findKNearest model features)
  in case configTaskType (modelConfig model) of
       Classification -> majorityVote pairs
       Regression     -> meanValue (map fst pairs)

-- | Предсказать метки для списка объектов с известными истинными метками.
predictAll
  :: Model
  -> [(Features, Label)]
  -> [PredictionResult]
predictAll model = map predictOne
  where
    predictOne (fs, actual) = PredictionResult
      { prFeatures  = fs
      , prPredicted = predict model fs
      , prActual    = actual
      }

-- | Найти k ближайших объектов обучающей выборки к запросу.
-- Возвращает список пар (объект, расстояние), отсортированных по расстоянию.
findKNearest :: Model -> Features -> [(DataPoint, Double)]
findKNearest model query =
  take k . sortBy (comparing snd) $ withDists
  where
    cfg         = modelConfig model
    trainingSet = modelTrainingSet model
    k           = configK cfg
    metric      = configMetric cfg
    withDists   =
      map (\dp -> (dp, computeDistance metric query (dpFeatures dp)))
          trainingSet

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
meanValue []  = 0.0
meanValue ls  = sum ls / fromIntegral (length ls)

-- | Классификация голосованием большинства среди соседей.
-- При равенстве голосов побеждает класс с ближайшим соседом.
majorityVote :: [(Label, Double)] -> Label
majorityVote [] = 0.0
majorityVote pairs =
  case sortBy cmp tied of
    []          -> 0.0
    ((lbl, _):_) -> lbl
  where
    labels   = map fst pairs
    unique   = foldr insertUniq [] labels
    counts   = map (\l -> (l, length (filter (== l) labels))) unique
    maxCount = maximum (map snd counts)
    tied     = filter (\(_, c) -> c == maxCount) counts
    minDist l = minimum [d | (lbl, d) <- pairs, lbl == l]
    cmp (l1, _) (l2, _) = compare (minDist l1) (minDist l2)
    insertUniq x acc = if x `elem` acc then acc else x : acc
