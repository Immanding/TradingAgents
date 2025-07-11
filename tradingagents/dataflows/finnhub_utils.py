import json
import os


def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, period=None):
    """
    获取保存在磁盘上的finnhub数据。
    参数:
        start_date (str): 开始日期，格式为YYYY-MM-DD。
        end_date (str): 结束日期，格式为YYYY-MM-DD。
        data_type (str): 从finnhub获取的数据类型。可以是insider_trans、SEC_filings、news_data、insider_senti或fin_as_reported。
        data_dir (str): 数据保存的目录。
        period (str): 默认为none，如果指定了period，应该是annual或quarterly。
    """

    if period:
        data_path = os.path.join(
            data_dir,
            "finnhub_data",
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, "finnhub_data", data_type, f"{ticker}_data_formatted.json"
        )

    data = open(data_path, "r")
    data = json.load(data)

    # 按日期范围过滤键（日期，格式为YYYY-MM-DD的字符串）
    filtered_data = {}
    for key, value in data.items():
        if start_date <= key <= end_date and len(value) > 0:
            filtered_data[key] = value
    return filtered_data
