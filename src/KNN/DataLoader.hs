{-# OPTIONS_GHC -Wall #-}

-- | Загрузка и разбор наборов данных из текстовых файлов.
--
-- Формат файла: числа, разделённые запятыми или пробелами, по одному объекту
-- на строку; последний столбец — метка класса или целевое значение.
module KNN.DataLoader
  ( loadDataset
  , parseDataset
  , selectColumns
  , LoadResult (..)
  , DatasetInfo (..)
  , datasetInfo
  , checkDims
  ) where

import Data.List (nub, sort)

import KNN.Types (DataPoint (..), TrainingSet)

-- | Результат загрузки набора данных.
data LoadResult = LoadResult
  { loadedPoints :: TrainingSet
  -- ^ Успешно разобранные объекты
  , loadWarnings :: [String]
  -- ^ Предупреждения о пропущенных строках
  } deriving (Show, Eq)

-- | Сводная информация о загруженном наборе данных.
data DatasetInfo = DatasetInfo
  { infoNumPoints   :: Int
  -- ^ Количество объектов (строк)
  , infoNumFeatures :: Int
  -- ^ Количество признаков на объект (столбцов без метки)
  } deriving (Show, Eq)

-- | Загрузить набор данных из файла.
-- Одиночные ошибки строк — предупреждения; >50% плохих строк — фатально.
loadDataset :: FilePath -> IO (Either String LoadResult)
loadDataset path = do
  content <- readFile path
  return (parseDataset content)

-- | Разобрать набор данных из текстового содержимого.
-- Строки с ошибками пропускаются; если их >50% — возвращает Left.
parseDataset :: String -> Either String LoadResult
parseDataset content =
  let rawLines  = filter (not . null) (lines content)
      numbered  = zip [1 :: Int ..] rawLines
      results   = map (\(n, l) -> (n, parseLine l)) numbered
      good      = [dp  | (_, Right dp)  <- results]
      bad       = [(n, e) | (n, Left e) <- results]
      total     = length rawLines
      badCount  = length bad
      warns     = map formatWarn bad
  in if total == 0
       then Left "Файл не содержит данных"
       else if badCount * 2 > total
         then Left
               ( "Слишком много некорректных строк: "
                 ++ show badCount ++ " из " ++ show total )
         else case checkDimConsistency good of
                Left err -> Left err
                Right () -> Right LoadResult
                  { loadedPoints = good
                  , loadWarnings = warns
                  }
  where
    formatWarn (n, e) = "Строка " ++ show n ++ " пропущена: " ++ e

-- | Проверить что все объекты выборки имеют одинаковое число признаков.
checkDimConsistency :: TrainingSet -> Either String ()
checkDimConsistency []             = Right ()
checkDimConsistency (first : rest) =
  let dim     = length (dpFeatures first)
      badRows = filter (\dp -> length (dpFeatures dp) /= dim) rest
  in case badRows of
       [] -> Right ()
       _  -> Left ( "Несовпадение числа признаков в выборке: ожидалось "
                    ++ show dim ++ " признаков в каждой строке" )

-- | Разобрать одну строку в объект DataPoint.
parseLine :: String -> Either String DataPoint
parseLine line =
  case mapM readDouble (splitOnSep line) of
    Left err  -> Left err
    Right []  -> Left ("Пустая строка: " ++ show line)
    Right [_] -> Left
      ("Строка содержит только одно значение, нужны признаки и метка: "
       ++ show line)
    Right vs  ->
      Right DataPoint { dpFeatures = init vs, dpLabel = last vs }

-- | Разбить строку по пробелам или запятым.
splitOnSep :: String -> [String]
splitOnSep s =
  case dropWhile isSep s of
    [] -> []
    s' -> let (token, rest) = break isSep s'
          in token : splitOnSep rest
  where
    isSep c = c == ',' || c == ' ' || c == '\t'

-- | Безопасный разбор вещественного числа.
readDouble :: String -> Either String Double
readDouble s =
  case reads s of
    [(v, "")] -> Right v
    _         -> Left ("Невозможно разобрать число: " ++ s)

-- | Оставить только указанные столбцы признаков
-- Фатальная ошибка если индекс выходит за границы.
selectColumns :: [Int] -> TrainingSet -> Either String TrainingSet
selectColumns cols ts =
  case ts of
    [] -> Right []
    (first : _) ->
      let dim        = length (dpFeatures first)
          cols'      = sort (nub cols)
          outOfRange = filter (\c -> c < 0 || c >= dim) cols'
      in case outOfRange of
           (c : _) -> Left
             ( "Индекс столбца " ++ show c
               ++ " выходит за границы (доступно " ++ show dim
               ++ " признаков, индексы 0–" ++ show (dim - 1) ++ ")" )
           [] -> Right (map (pickCols cols') ts)
  where
    pickCols cols' dp =
      dp { dpFeatures = [dpFeatures dp !! c | c <- cols'] }

-- | Проверить совместимость числа признаков между двумя выборками.
checkDims :: DatasetInfo -> DatasetInfo -> Either String ()
checkDims train test
  | infoNumFeatures train == infoNumFeatures test = Right ()
  | otherwise = Left
      ( "Несовпадение числа признаков: обучающая выборка имеет "
        ++ show (infoNumFeatures train)
        ++ " признаков, тестовая — "
        ++ show (infoNumFeatures test) )

-- | Вычислить сводную информацию о наборе данных.
datasetInfo :: TrainingSet -> DatasetInfo
datasetInfo []        =
  DatasetInfo { infoNumPoints = 0, infoNumFeatures = 0 }
datasetInfo (x : xs) =
  DatasetInfo
    { infoNumPoints   = 1 + length xs
    , infoNumFeatures = length (dpFeatures x)
    }
