import abc
from functools import partial

import equinox as eqx
import equinox.internal as eqxi

# from integrax.ad_jvp import implicit_jvp
from integrax.ad import leibniz_integration_rule


class AbstractAdjoint(eqx.Module):
    @abc.abstractmethod
    def apply(self):
        pass


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    checkpoints: int | None = None

    def apply(self, fn_primal, inputs, max_steps):
        while_loop = partial(
            eqxi.while_loop,
            kind="checkpointed",
            checkpoints=self.checkpoints,
            max_steps=max_steps,
        )
        return fn_primal(inputs + (while_loop,))


class ImplicitAdjoint(AbstractAdjoint):
    def apply(self, fn_primal, inputs, max_steps):
        while_loop = partial(eqxi.while_loop, kind="lax", max_steps=max_steps)
        fn, method, lower, upper, args, options = inputs
        return leibniz_integration_rule(
            (fn, lower, upper, args),
            fn_primal=fn_primal,
            method=method,
            options=options,
            while_loop=while_loop,
        )
        # return implicit_jvp(fn_primal, inputs + (while_loop,))
