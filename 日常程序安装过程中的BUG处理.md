# 日常程序安装过程中的BUG处理

## Redis

安装后无法启动
`Failed to start redis.servive.service: Unit not found.`
`Could not connect to Redis at 127.0.0.1:6379: Connection refused`
最后是通过:
`sudo systemctl start redis`
先启动服务完成的。 `o(╯□╰)o`