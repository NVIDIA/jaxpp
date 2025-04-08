# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as it
from collections import defaultdict
from collections.abc import Callable
from functools import partial

import jax
import jax.core
import tensorboard.compat.proto as tbp
from graphviz import Digraph
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# Tensorboard constants
_TB_TS_OUTPUT_SHAPES_KEY = "_output_shapes"
_TB_TS_XLACLUSTER_KEY = "_XlaCluster"
_TB_CONST_OP = "Const"

# Jax constants
_JAX_CALL_JAXPR_KEY = "call_jaxpr"
_JAX_CALL_JAXPR_NAME_KEY = "name"

AttrValue = tbp.attr_value_pb2.AttrValue
NodeDef = tbp.node_def_pb2.NodeDef


def mk_shape_attr(a: jax.core.ShapedArray) -> tuple[str, tbp.attr_value_pb2.AttrValue]:
    # https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html
    # the true shape of an array is `.shape` + `.named_shape`
    TensorShapeProto = tbp.tensor_shape_pb2.TensorShapeProto
    Dim = TensorShapeProto.Dim

    dim = jax.util.safe_map(lambda size: Dim(size=size), a.shape) + jax.util.safe_map(
        lambda name_size: Dim(name=name_size[0], size=name_size[1]),
        a.named_shape.items(),
    )
    return (
        _TB_TS_OUTPUT_SHAPES_KEY,
        AttrValue(list=AttrValue.ListValue(shape=[TensorShapeProto(dim=dim)])),
    )


def to_graph(
    jaxpr: jax.core.Jaxpr,
    curr_namespace="",
    sub: dict[jax.core.Var, str] | None = None,
) -> tbp.graph_pb2.GraphDef:
    if sub is None:
        sub = {}

    def namespaced(s: str) -> str:
        if curr_namespace != "":
            return f"{curr_namespace}/{s}"
        return s

    nodes = []
    for xvar in it.chain(jaxpr.constvars, jaxpr.invars, jaxpr.outvars):
        attr, val = mk_shape_attr(jax.core.raise_to_shaped(xvar.aval))
        if xvar not in sub:
            nodes.append(
                NodeDef(
                    name=f"{sub.get(xvar, None) or namespaced(str(xvar))}",
                    op="Parameter",
                    device="/device:MESH:0",
                    attr={
                        _TB_TS_XLACLUSTER_KEY: AttrValue(
                            s=bytes("foo.org:3333", "utf-8")
                        ),
                        attr: val,
                    },
                )
            )

    for idx, eqn in enumerate(jaxpr.eqns):
        is_const = [isinstance(invar, jax.core.Literal) for invar in eqn.invars]
        invar_names = [
            namespaced(f"{idx}.const_{cidx}")
            if literal
            else f"{sub.get(invar, None) or namespaced(str(invar))}"
            for cidx, (literal, invar) in enumerate(
                jax.util.safe_zip(is_const, eqn.invars)
            )
        ]

        for literal, invar in jax.util.safe_zip(is_const, invar_names):
            if literal:
                nodes.append(NodeDef(name=invar, op=f"{_TB_CONST_OP}"))

        subjaxpr = eqn.params.get(_JAX_CALL_JAXPR_KEY, None)
        if subjaxpr is not None:
            if isinstance(subjaxpr, jax.core.ClosedJaxpr):
                subjaxpr = subjaxpr.jaxpr
            subjaxpr: jax.core.Jaxpr
            name = eqn.params.get(_JAX_CALL_JAXPR_NAME_KEY, None)
            if name is not None:
                name = namespaced(f"{name}.{eqn.primitive!s}.{idx}")
            else:
                name = namespaced(f"{eqn.primitive!s}.{idx}")
            recur_sub = dict(
                jax.util.safe_zip(
                    list(it.chain(subjaxpr.invars, subjaxpr.outvars)),
                    list(it.chain(invar_names, eqn.outvars)),
                )
            )
            rec = to_graph(subjaxpr, name, recur_sub)
            nodes.extend(rec.node)
        else:
            for outvar in eqn.outvars:
                attr, val = mk_shape_attr(jax.core.raise_to_shaped(outvar.aval))

                nodes.append(
                    NodeDef(
                        name=f"{sub.get(outvar, None) or namespaced(str(outvar))}",
                        op=f"{eqn.primitive}",
                        input=invar_names,
                        device="/device:MESH:0",
                        attr={
                            _TB_TS_XLACLUSTER_KEY: AttrValue(
                                s=bytes("foo.org:3333", "utf-8")
                            ),
                            attr: val,
                        },
                    )
                )

    graph = tbp.graph_pb2.GraphDef(node=nodes)
    return graph


def SummaryWriter(
    logdir="tb_logs", max_queue_size=10, flush_secs=120, filename_suffix=""
) -> EventFileWriter:
    return EventFileWriter(
        logdir=logdir,
        max_queue_size=max_queue_size,
        flush_secs=flush_secs,
        filename_suffix=filename_suffix,
    )


def write_graph(writer: EventFileWriter, graph_def: tbp.graph_pb2.GraphDef):
    event = tbp.event_pb2.Event(graph_def=graph_def.SerializeToString())
    writer.add_event(event)


styles = {
    "const": dict(style="filled", color="goldenrod1"),
    "invar": dict(color="mediumspringgreen", style="filled"),
    "outvar": dict(style="filled,dashed", fillcolor="indianred1", color="black"),
    "op_node": dict(shape="box", color="lightskyblue", style="filled"),
    "intermediate": dict(style="filled", color="cornflowerblue"),
}


def _jaxpr_graph(jaxpr: jax.core.Jaxpr, var_label=Callable[[jax.core.Var], str]):
    if var_label is None:

        def mt(var):
            return ""

        var_label = mt
    id_names = (f"id{id}" for id in it.count())

    graph = Digraph(engine="dot")
    graph.attr(rankdir="LR")
    graph.attr(size="6,10!")

    subgraphs = defaultdict(list)

    outvars_set = set(jaxpr.outvars)

    def mk_jaxpr_var_node(v, style):
        graph.node(
            str(v),
            f"{var_label(v)} {jax.core.raise_to_shaped(v.aval).str_short()}",
            styles[style],
        )

    jax.util.safe_map(partial(mk_jaxpr_var_node, style="const"), jaxpr.constvars)
    jax.util.safe_map(partial(mk_jaxpr_var_node, style="invar"), jaxpr.invars)
    jax.util.safe_map(partial(mk_jaxpr_var_node, style="outvar"), jaxpr.outvars)

    for eqn in jaxpr.eqns:
        if eqn.primitive.multiple_results:
            id_name = next(id_names)
            name = str(eqn.primitive)
            if "name" in eqn.params and "task_type" in eqn.params:
                name = f"{name} {eqn.params['name']} {eqn.b['task_type']}"
                subgraphs[eqn.params["name"]].append(id_name)
            graph.node(id_name, name, styles["op_node"])
            for v in eqn.outvars:
                style = "intermediate" if v not in outvars_set else "outvar"
                graph.node(
                    str(v), f"{var_label(v)} {v}: {v.aval.str_short()}", styles[style]
                )
                graph.edge(id_name, str(v))
            for v in eqn.invars:
                if isinstance(v, jax.core.Var):
                    graph.edge(str(v), id_name)
        else:
            (outv,) = eqn.outvars
            style = "op_node" if outv not in outvars_set else "outvar"
            graph.node(
                str(outv),
                f"{var_label(outv)} {outv}: {eqn.primitive!s}",
                styles[style],
            )
            for v in eqn.invars:
                if isinstance(v, jax.core.Var):
                    graph.edge(str(v), str(outv))

    for name, nodes in subgraphs.items():
        with graph.subgraph(name=name) as subg:
            subg.attr(rank="same")
            for node in nodes:
                subg.node(node)
    return graph
