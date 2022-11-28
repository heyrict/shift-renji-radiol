#!/usr/bin/env python3
# Copyright 2010-2022 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates a shift scheduling problem and solves it."""

import toml
from typing import Any

from absl import app
from absl import flags

from ortools.sat.python import cp_model
from google.protobuf import text_format

FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'config.toml',
                    'Config toml file')
flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:30.0',
                    'Sat solver parameters.')


def find_index(seq: list[Any], value, key="id", seqname="sequence", forced=True):
    try:
        index = [x[key] for x in seq].index(value)
    except:
        if forced:
            raise ValueError(f"Key `{key}={value}` not found in {seqname}")
        else:
            return -1
    return index


def negated_bounded_span(works, start, length):
    """
    Filters an isolated sub-sequence of variables assined to True.
    Extract the span of Boolean variables [start, start + length), negate them,
    and if there is variables to the left/right of this span, surround the span by
    them in non negated form.
    Args:
      works: a list of variables to extract the span from.
      start: the start to the span.
      length: the length of the span.
    Returns:
      a list of variables which conjunction will be false if the sub-list is
      assigned to True, and correctly bounded by variables assigned to False,
      or by the start or end of works.
    """

    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence


def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """
    Sequence constraint on true variables with soft and hard bounds.
    This constraint look at every maximal contiguous sequence of variables
    assigned to true. If forbids sequence of length < hard_min or > hard_max.
    Then it creates penalty terms if the length is < soft_min or > soft_max.
    Args:
      model: the sequence constraint is built on this model.
      works: a list of Boolean variables.
      hard_min: any sequence of true variables must have a length of at least
        hard_min.
      soft_min: any sequence should have a length of at least soft_min, or a
        linear penalty on the delta will be added to the objective.
      min_cost: the coefficient of the linear penalty if the length is less than
        soft_min.
      soft_max: any sequence should have a length of at most soft_max, or a linear
        penalty on the delta will be added to the objective.
      hard_max: any sequence of true variables must have a length of at most
        hard_max.
      max_cost: the coefficient of the linear penalty if the length is more than
        soft_max.
      prefix: a base name for penalty literals.
    Returns:
      a tuple (variables_list, coefficient_list) containing the different
      penalties created by the sequence constraint.
    """

    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': under_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': over_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients


def add_soft_sum_constraint(model, works, hard_min, soft_min, min_cost,
                            soft_max, hard_max, max_cost, prefix):
    """Sum constraint with soft and hard bounds.
  This constraint counts the variables assigned to true from works.
  If forbids sum < hard_min or > hard_max.
  Then it creates penalty terms if the sum is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a sum of at least
      hard_min.
    soft_min: any sequence should have a sum of at least soft_min, or a linear
      penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the sum is less than
      soft_min.
    soft_max: any sequence should have a sum of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a sum of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the sum is more than
      soft_max.
    prefix: a base name for penalty variables.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_variables = []
    cost_coefficients = []
    sum_var = model.NewIntVar(hard_min, hard_max, '')
    # This adds the hard constraints on the sum.
    model.Add(sum_var == sum(works))

    # Penalize sums below the soft_min target.
    if soft_min > hard_min and min_cost > 0:
        delta = model.NewIntVar(-len(works), len(works), '')
        model.Add(delta == soft_min - sum_var)
        # TODO(user): Compare efficiency with only excess >= soft_min - sum_var.
        excess = model.NewIntVar(0, 7, prefix + ': under_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(min_cost)

    # Penalize sums above the soft_max target.
    if soft_max < hard_max and max_cost > 0:
        delta = model.NewIntVar(-7, 7, '')
        model.Add(delta == sum_var - soft_max)
        excess = model.NewIntVar(0, 7, prefix + ': over_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(max_cost)

    return cost_variables, cost_coefficients


def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""
    with open(FLAGS.config, 'r') as f:
        config = toml.decoder.load(f)

    # Data
    shifts: list[Any] = config["Shift"]
    employees: list[Any] = config["Employee"]
    fixed_assignments: list[Any] = config.get("FixedAssignment", [])
    requests: list[Any] = config.get("Request", [])
    shift_constraints: list[Any] = config.get("ShiftConstraint", [])
    penalized_transitions: list[Any] = config.get("PenalizedTransition", [])
    profiles: list[Any] = config.get("Profile", [])

    num_employees = len(employees)
    num_shifts = len(shifts)
    num_weeks = config.get("General", {}).get("num_weeks", 3)
    num_days = num_weeks * 7

    model = cp_model.CpModel()

    work = {}
    for e in range(num_employees):
        for s in range(num_shifts):
            for d in range(num_days):
                work[e, s, d] = model.NewBoolVar('work%i_%i_%i' % (e, s, d))

    # Linear terms of the objective in a minimization context.
    obj_int_vars = []
    obj_int_coeffs = []
    obj_bool_vars = []
    obj_bool_coeffs = []

    # Exactly one shift per day.
    for e in range(num_employees):
        for d in range(num_days):
            model.AddExactlyOne(work[e, s, d] for s in range(num_shifts))

    # Fixed assignments.
    for fa in fixed_assignments:
        e = find_index(employees, fa["employee"], seqname="employees")
        s = find_index(shifts, fa["shift"], seqname="shifts")
        d = fa["day"]
        model.Add(work[e, s, d] == 1)

    # Employee requests
    for rq in requests:
        e = find_index(employees, rq["employee"], seqname="employees")
        s = find_index(shifts, rq["shift"], seqname="shifts")
        d = rq["day"]
        cost = rq["penalty"]
        obj_bool_vars.append(work[e, s, d])
        obj_bool_coeffs.append(cost)

    # Shift constraints
    for ct in shift_constraints:
        shift = find_index(shifts, ct["shift"], seqname="shifts")
        hard_min = ct["hard_min"]
        hard_max = ct["hard_max"]
        soft_min = ct["soft_min"]
        soft_max = ct["soft_max"]
        min_cost = ct["min_penalty"]
        max_cost = ct["max_penalty"]
        for e, employee in enumerate(employees):
            works = [work[e, shift, d] for d in range(num_days)]
            variables, coeffs = add_soft_sequence_constraint(
                model, works, hard_min, soft_min, min_cost, soft_max, hard_max,
                max_cost,
                'shift_constraint(employee %s, shift %s)' % (employee["name"], shifts[shift]["name"]))
            obj_bool_vars.extend(variables)
            obj_bool_coeffs.extend(coeffs)

    # Weekly sum constraints
    for e, employee in enumerate(employees):
        profile_id = find_index(
            profiles, employee["profile_id"], seqname="profiles")
        profile = profiles[profile_id]
        for s, shift in enumerate(shifts):
            ctid = find_index(profile["shifts"], shift["id"], key="shift_id",
                              seqname=f"profile({profile['id']})/shifts", forced=False)
            if ctid == -1:
                for d in range(num_days):
                    model.AddBoolOr(work[e, s, d].Not())
            else:
                ct = profile["shifts"][ctid]
                hard_min = ct["hard_min"]
                hard_max = ct["hard_max"]
                soft_min = ct["soft_min"]
                soft_max = ct["soft_max"]
                min_cost = ct["min_penalty"]
                max_cost = ct["max_penalty"]
                for w in range(num_weeks):
                    works = [work[e, s, d + w * 7] for d in range(7)]
                    variables, coeffs = add_soft_sum_constraint(
                        model, works, hard_min, soft_min, min_cost, soft_max,
                        hard_max, max_cost,
                        'weekly_sum_constraint(employee %s, shift %s, week %i)' %
                        (employee["name"], shift["name"], w))
                    obj_int_vars.extend(variables)
                    obj_int_coeffs.extend(coeffs)

    # Penalized transitions
    for pt in penalized_transitions:
        previous_shift = find_index(shifts, pt["shift_prev"], seqname="shifts")
        next_shift = find_index(shifts, pt["shift_next"], seqname="shifts")
        cost = pt["penalty"]
        for e, employee in enumerate(employees):
            for d in range(num_days - 1):
                transition = [
                    work[e, previous_shift, d].Not(), work[e, next_shift,
                                                           d + 1].Not()
                ]
                if cost == 0:
                    model.AddBoolOr(transition)
                else:
                    trans_var = model.NewBoolVar(
                        'transition (employee=%s, day=%i)' % (employee["name"], d))
                    transition.append(trans_var)
                    model.AddBoolOr(transition)
                    obj_bool_vars.append(trans_var)
                    obj_bool_coeffs.append(cost)

    # Penalize double work or double rest at weekend
    # Assume shift id = 0 for rest
    weekend_full_work_penalty = config.get(
        "General", {}).get("weekend_full_work_penalty", 0)
    if weekend_full_work_penalty:
        for e, employee in enumerate(employees):
            d = 5
            profile_id = find_index(
                profiles, employee["profile_id"], seqname="profiles")
            profile = profiles[profile_id]
            shift_ids = [ct["shift_id"] for ct in profile["shifts"]]
            shift_list = [find_index(shifts, sid,
                                     seqname=f"shifts") for sid in shift_ids]
            shift_work_list = [x for x in shift_list if x != 0]

            # Double work
            for s1 in shift_work_list:
                for s2 in shift_work_list:
                    for w in range(num_weeks):
                        works = [work[e, s1, w * 7 + d].Not(),
                                 work[e, s2, w * 7 + d + 1].Not()]
                        lit = model.NewBoolVar('weekend_full_work(employee=%s, week=%s, sat=%s, sun=%s)' % (
                            employee["name"], w, shifts[s1]["name"], shifts[s2]["name"]))
                        works.append(lit)
                        model.AddBoolOr(works)
                        obj_bool_vars.append(lit)
                        obj_bool_coeffs.append(weekend_full_work_penalty)

    # Cover constraints
    for s, shift in enumerate(shifts):
        demands = shift.get("demands")
        penalty_type = shift.get("penalty_type", "squared")
        if demands is None:
            continue
        cost = shift.get("excess_penalty", 2)
        if isinstance(cost, int):
            cost = [cost] * 7
        demands *= num_weeks
        # FIXME: First day is always monday here
        for d, demand in enumerate(demands):
            works = [work[e, s, d] for e in range(num_employees)]
            worked = model.NewIntVar(demand, num_employees, '')
            model.Add(worked == sum(works))

            weekday = d % 7
            over_penalty = cost[weekday]
            if over_penalty > 0:
                name = 'excess_demand(shift=%s, day=%i)' % (shift["name"], d)
                excess = model.NewIntVar(0, num_employees - demand, name)
                model.Add(excess == worked - demand)
                if penalty_type == "linear":
                    obj_int_vars.append(excess)
                    obj_int_coeffs.append(over_penalty)
                elif penalty_type == "squared":
                    name = 'excess_demand_sq(shift=%s, day=%i)' % (
                        shift["name"], d)
                    excess_sq = model.NewIntVar(
                        0, (num_employees - demand) ** 2, name)
                    model.AddMultiplicationEquality(
                        excess_sq, [excess, excess])
                    obj_int_vars.append(excess_sq)
                    obj_int_coeffs.append(over_penalty)
                else:
                    raise ValueError(f"Invalid value: Shift/{shift['id']}/penalty_type "
                                     f"== {penalty_type}"
                                     )

    # Objective
    model.Minimize(
        sum(obj_bool_vars[i] * obj_bool_coeffs[i]
            for i in range(len(obj_bool_vars))) +
        sum(obj_int_vars[i] * obj_int_coeffs[i]
            for i in range(len(obj_int_vars))))

    if output_proto:
        print('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)  # type: ignore
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.Solve(model, solution_printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print()
        header = '           '
        for w in range(num_weeks):
            header += 'Mo Tu We Th Fr Sa Su '
        print(header)
        for e, employee in enumerate(employees):
            schedule = ''
            for d in range(num_days):
                for s in range(num_shifts):
                    if solver.BooleanValue(work[e, s, d]):
                        schedule += shifts[s]["name"] + ' '
            print('worker %s: %s' % (employee["name"], schedule))
        print()
        print('Penalties:')
        for i, var in enumerate(obj_bool_vars):
            if solver.BooleanValue(var):
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    print('  %s violated, penalty=%i' % (var.Name(), penalty))
                else:
                    print('  %s fulfilled, gain=%i' % (var.Name(), -penalty))
            else:
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    print('  %s not violated, penalty=%i' %
                          (var.Name(), penalty))
                else:
                    print('  %s not fulfilled, gain=%i' %
                          (var.Name(), -penalty))

        for i, var in enumerate(obj_int_vars):
            if solver.Value(var) > 0:
                print('  %s violated by %i, linear penalty=%i' %
                      (var.Name(), solver.Value(var), obj_int_coeffs[i]))

    print()
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())


def main(_):
    solve_shift_scheduling(FLAGS.params, FLAGS.output_proto)


if __name__ == '__main__':
    app.run(main)
