import weightwatcher as ww
import pprint

from models.EfficientNet import EfficientNetVariant
from models.ModelArchitecture import ModelArchitecture
from services.ModelWrapperService import ModelWrapperService


def main():
    efficientnet_b0_wrapper = ModelWrapperService.get_model_wrapper(ModelArchitecture.EfficientNet, EfficientNetVariant.B0)
    watcher = ww.WeightWatcher(model=efficientnet_b0_wrapper.model)
    details = watcher.analyze(plot=True, savefig=True)
    summary = watcher.get_summary(details)
    pprint.pprint(summary)


if __name__ == '__main__':
    main()
