import logging

logging.basicConfig(level=logging.WARNING, filename='logs/ml_errors.log')


def safe_execute(func, *args, error_msg="Ошибка при выполнении", **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"{error_msg}: {str(e)}", exc_info=True)
        print(f"⚠️ {error_msg}: {str(e)}")
        raise
