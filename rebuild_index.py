#!/usr/bin/env python3
"""重建数据库索引并压缩数据库

使用场景：
1. 优化数据库大小
2. 应用新的预处理逻辑
3. 清理碎片空间
"""

import sqlite3
from pathlib import Path
from claude_viewer import DB_PATH, init_db, scan_session_files, index_file

def rebuild_database():
    """完全重建数据库"""
    print(f"数据库路径: {DB_PATH}")

    # 备份原数据库
    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix('.sqlite3.backup')
        print(f"备份原数据库到: {backup_path}")
        import shutil
        shutil.copy2(DB_PATH, backup_path)

    # 删除旧数据库
    if DB_PATH.exists():
        print("删除旧数据库...")
        DB_PATH.unlink()

    # 创建新数据库
    print("创建新数据库...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    # 重新索引所有文件
    print("重新索引会话文件...")
    total = 0
    for source, project, path in scan_session_files():
        print(f"  索引: {source}/{project}/{path.name}")
        index_file(conn, source, project, path)
        total += 1

    conn.commit()

    # VACUUM 压缩
    print("压缩数据库...")
    conn.execute("VACUUM")

    conn.close()

    # 显示结果
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n完成！")
    print(f"  索引文件数: {total}")
    print(f"  数据库大小: {size_mb:.1f} MB")

if __name__ == "__main__":
    rebuild_database()
