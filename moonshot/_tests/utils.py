def round_results(results_dict_or_list, n=6):
    """
    Rounds the values in results_dict, which can be scalars or
    lists.
    """
    def round_if_can(value):
        try:
            return round(value, n)
        except TypeError:
            return value

    if isinstance(results_dict_or_list, dict):
        for key, value in results_dict_or_list.items():
            if isinstance(value, list):
                results_dict_or_list[key] = [round_if_can(v) for v in value]
            else:
                results_dict_or_list[key] = round_if_can(value)
        return results_dict_or_list
    else:
        return [round_if_can(value) for value in results_dict_or_list]
