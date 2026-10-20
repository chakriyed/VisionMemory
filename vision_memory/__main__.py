import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        from .chat import main
        main(sys.argv[2:])
    else:
        from .app import main
        main()
