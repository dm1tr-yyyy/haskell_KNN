{-# OPTIONS_GHC -Wall #-}

-- Точка входа: разбор аргументов командной строки, запуск
-- k-NN и вывод читаемого отчёта.
--
-- Использование:
--   stack run -- <обучение> <тест> <модель> [k] [метрика] [задача] [столбцы]
--
-- Допустимые значения метрики : Euclidean | Manhattan | Chebyshev
-- Допустимые значения задачи  : Classification | Regression
-- Формат столбцов             : 0,1,3  (индексы 0-based через запятую)
-- По умолчанию: k=3, метрика=Euclidean, задача=Classification, все столбцы
module Main (main) where

import System.Environment (getArgs)

import Lib

-- | Точка входа в программу.
main :: IO ()
main = do
  args <- getArgs
  case args of
    (trainFile : testFile : modelFile : rest) ->
      runPipeline trainFile testFile modelFile rest
    [modelFile, testFile] ->
      runWithLoadedModel modelFile testFile
    _ ->
      putStrLn usage
