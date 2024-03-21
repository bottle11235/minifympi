import click

@click.command()
@click.option('-n', '--nprocs', default=-1, help='The person to greet.')
def run(nprocs):        
    click.echo(f'Hello, {nprocs}!')

if __name__ == '__main__':
    # 解析命令行参数
    import sys
    args = sys.argv[1:]
    ctx = run.make_context('run', args)
    run.invoke(ctx)

    # 后续代码
    print("Continuing with the rest of the code...")
