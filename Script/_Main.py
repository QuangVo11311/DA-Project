import DataPreprocessing as DPre
import DataAnalysis as DAna

def MainStep():
    OrderDf, ProductDf = DPre.PreProcessing()
    DAna.Analysis(ProductDf, OrderDf)

if __name__ == '__main__':
    MainStep()