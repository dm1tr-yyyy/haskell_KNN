{-# OPTIONS_GHC -Wall #-}

-- | Загрузка и разбор наборов данных из текстовых файлов.
--
-- Формат файла: числа, разделённые запятыми, по одному объекту на строку;
-- последний столбец — метка класса или целевое значение.
module KNN.DataLoader
  ( loadDataset
  , parseDataset
  , DatasetInfo (..)
  , datasetInfo
  ) where

import KNN.Types (DataPoint (..), TrainingSet)

-- | Сводная информация о загруженном наборе данных.
data DatasetInfo = DatasetInfo
  { infoNumPoints   :: Int
  -- ^ Количество объектов (строк)
  , infoNumFeatures :: Int
  -- ^ Количество признаков на объект (столбцов без метки)
  } deriving (Show, Eq)

-- | Загрузить набор данных из файла.
-- Возвращает Left с сообщением об ошибке при неудаче.
loadDataset :: FilePath -> IO (Either String TrainingSet)
loadDataset path = do
  content <- readFile path
  return (parseDataset content)

-- | Разобрать набор данных из текстового содержимого.
-- Возвращает Left с сообщением об ошибке, если хотя бы одна строка некорректна.
parseDataset :: String -> Either String TrainingSet
parseDataset content =
  mapM parseLine (filter (not . null) (lines content))

-- | Разобрать одну строку с числами (разделитель — пробел или запятая)
-- в объект DataPoint.
-- Возвращает Left с сообщением об ошибке, если строка содержит менее 2 значений.
parseLine :: String -> Either String DataPoint
parseLine line =
  case mapM readDouble (splitOnSep line) of
    Left err  -> Left err
    Right []  -> Left ("Пустая строка: " ++ show line)
    Right [_] -> Left
      ("Строка содержит только одно значение, нужны признаки и метка: "
       ++ show line)
    Right vs  ->
      let features = init vs
          label    = last vs
      in Right DataPoint { dpFeatures = features, dpLabel = label }

-- | Разбить строку по пробелам или запятым.
splitOnSep :: String -> [String]
splitOnSep s =
  case dropWhile isSep s of
    [] -> []
    s' -> let (token, rest) = break isSep s'
          in token : splitOnSep rest
  where
    isSep c = c == ',' || c == ' ' || c == '\t'

-- | Безопасный разбор вещественного числа; возвращает Left при ошибке.
readDouble :: String -> Either String Double
readDouble s =
  case reads s of
    [(v, "")] -> Right v
    _         -> Left ("Невозможно разобрать число: " ++ show s)

-- | Вычислить сводную информацию о наборе данных.
datasetInfo :: TrainingSet -> DatasetInfo
datasetInfo [] =
  DatasetInfo { infoNumPoints = 0, infoNumFeatures = 0 }
datasetInfo (x : xs) =
  DatasetInfo
    { infoNumPoints   = length $ x : xs
    , infoNumFeatures = length (dpFeatures x)
    }