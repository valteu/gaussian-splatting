from create_colmaps import create_colmaps
from train_colmaps import train_colmaps
from render_all import render_all
from metric_all import metric_all


def main():
    create_colmaps()
    train_colmaps()
    render_all()
    metric_all()


if __name__ == "__main__":
    main()
