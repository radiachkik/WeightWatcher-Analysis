from dataclasses import dataclass


@dataclass
class AnalyseHardcodedModelsOptions:
    resume: bool
    include_untrained: bool
    results_directory: str


def main(options: AnalyseHardcodedModelsOptions):
    from application import AnalyzeService
    from hardcoded_models import register_hardcoded_models
    from models import configure_cpu, ModelQueryService
    from ww import WWResultRepository
    import logging

    logging.basicConfig(level=logging.DEBUG)
    configure_cpu()

    register_hardcoded_models(pretrained=True)
    if options.include_untrained:
        register_hardcoded_models(pretrained=False)

    model_query_service = ModelQueryService()
    model_wrappers = model_query_service.get_all()
    ww_result_repository = WWResultRepository(options.results_directory)
    AnalyzeService.recalculate_all_summaries(ww_result_repository)
    if options.resume:
        AnalyzeService.resume_analyzing(model_wrappers, ww_result_repository)
    else:
        AnalyzeService.analyze(model_wrappers, ww_result_repository)


def parse_arguments() -> AnalyseHardcodedModelsOptions:
    import argparse

    parser = argparse.ArgumentParser(description='Analyze hardcoded models')
    parser.add_argument(
        "-r",
        "--resume",
        action='store_true',
        help="Whether to resume the analysis of hardcoded models"
    )
    parser.add_argument(
        "-u",
        "--untrained",
        action='store_true',
        help="Whether to include untrained versions of the hardcoded models (additionally to the trained versions)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ww_results",
        help="Output folder to store the results in"
    )
    args = parser.parse_args()
    options = AnalyseHardcodedModelsOptions(
        resume=args.resume,
        include_untrained=args.untrained,
        results_directory=args.output
    )
    return options


if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
