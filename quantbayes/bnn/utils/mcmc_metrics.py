import arviz as az

from quantbayes import bnn


def evaluate_mcmc(model: bnn.Module) -> None:
    """
    Evaluate an MCMC model by computing common diagnostics.

    :param model: A model instance (subclass of Module) that has run inference.
    :raises ValueError: If the model does not have inference results.
    """
    if not hasattr(model, "inference") or model.inference is None:
        raise ValueError(
            "The model does not have inference results. Please run inference first!"
        )

    idata = az.from_numpyro(model.inference)
    waic_result = az.waic(idata)
    loo_result = az.loo(idata)
    rhat_result = az.rhat(idata)
    ess_result = az.ess(idata)

    print("WAIC:", waic_result)
    print("LOO:", loo_result)
    print("rhat:", rhat_result)
    print("ESS:", ess_result)
