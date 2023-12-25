'''
Truncated gaussian correction functions
from the bottom of page 4 of the TrueSkill paper
'''

from skills.numerics import Gaussian

def v_exceeds_margin_scaled(team_performance_difference, draw_margin, c):
    return v_exceeds_margin(team_performance_difference / c, draw_margin / c)

def v_exceeds_margin(team_performance_difference, draw_margin):
    denominator = Gaussian.cumulative_to(team_performance_difference - draw_margin)
    if (denominator < 2.22275874e-162):
        return -team_performance_difference + draw_margin
    return Gaussian.at(team_performance_difference - draw_margin) / denominator

def w_exceeds_margin_scaled(team_performance_difference, draw_margin, c):
    return w_exceeds_margin(team_performance_difference / c, draw_margin / c)

def w_exceeds_margin(team_performance_difference, draw_margin):
    denominator = Gaussian.cumulative_to(team_performance_difference - draw_margin)
    if denominator < 2.222758749e-162:
        if team_performance_difference < 0.0:
            return 1.0
        return 0.0
    v_win = v_exceeds_margin(team_performance_difference, draw_margin)
    return v_win * (v_win + team_performance_difference - draw_margin)

def v_within_margin_scaled(team_performance_difference, draw_margin, c):
    return v_within_margin(team_performance_difference / c, draw_margin / c)

def v_within_margin(team_performance_difference, draw_margin):
    team_performance_difference_abs = abs(team_performance_difference)
    denominator = (
        Gaussian.cumulative_to(draw_margin - team_performance_difference_abs) -
        Gaussian.cumulative_to(-draw_margin - team_performance_difference_abs))

    if denominator < 2.222758749e-162:
        if team_performance_difference < 0.0:
            return -team_performance_difference - draw_margin
        return -team_performance_difference + draw_margin

    numerator = (Gaussian.at(-draw_margin - team_performance_difference_abs) -
                 Gaussian.at(draw_margin - team_performance_difference_abs))

    if team_performance_difference < 0.0:
        return -numerator / denominator
    return numerator / denominator

def w_within_margin_scaled(team_performance_difference, draw_margin, c):
    return w_within_margin(team_performance_difference / c, draw_margin / c)

def w_within_margin(team_performance_difference, draw_margin):
    abs_team_performance_difference = abs(team_performance_difference)
    print(f"w_within_margin - team_performance_difference: {team_performance_difference}, draw_margin: {draw_margin}, abs_team_performance_difference: {abs_team_performance_difference}")

    # Calculating the upper and lower boundaries for performance.
    upper_bound = draw_margin - team_performance_difference
    lower_bound = -draw_margin - team_performance_difference

    # Calculating the Gaussian probability density for the boundaries.
    pdf_upper_bound = Gaussian.at(upper_bound)
    pdf_lower_bound = Gaussian.at(lower_bound)

    # Calculating the cumulative probabilities for the boundaries.
    cdf_upper_bound = Gaussian.cumulative_to(upper_bound)
    cdf_lower_bound = Gaussian.cumulative_to(lower_bound)

    # Computing the denominator as the difference between the cumulative probabilities.
    denominator = cdf_upper_bound - cdf_lower_bound

    # Handling cases where the denominator is very close to zero to avoid ZeroDivisionError.
    if abs(denominator) < 1e-10:
        return 0.0 if team_performance_difference < 0 else 1.0  # Depending on the team_performance_difference, return 0 or 1.

    # Calculating the numerator by multiplying the Gaussian probability densities with the boundaries.
    #numerator = pdf_upper_bound - pdf_lower_bound
    vt = v_within_margin(abs_team_performance_difference, draw_margin)
    numerator = (upper_bound * pdf_upper_bound - lower_bound * pdf_lower_bound)

    # Final value is the ratio of the numerator to the denominator.
    return vt ** 2 + numerator / denominator



