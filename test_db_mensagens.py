import oracledb
import os
import sys

# Configurações do Oracle Thick Mode
instant_client_path = "/home/rdt/CameraIA/instantclient_21_1"
oracle_wallet_path = "/home/rdt/CameraIA/DriveOracle"

os.environ['LD_LIBRARY_PATH'] = f"{instant_client_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['TNS_ADMIN'] = oracle_wallet_path

try:
    oracledb.init_oracle_client(lib_dir=instant_client_path)
    print("Oracle Thick Mode inicializado.")
except Exception as e:
    print(f"Erro ao inicializar Thick Mode: {e}")

def test_db():
    try:
        conn = oracledb.connect(
            user="mensagem",
            password="crbsAcs@2026",
            dsn="imaculado",
            config_dir=oracle_wallet_path,
            wallet_location=oracle_wallet_path
        )
        print("Conexão estabelecida com sucesso!")
        
        cursor = conn.cursor()
        
        # 1. Tenta inserir uma mensagem de teste
        test_phone = "5511999999999"
        test_msg = "TESTE MANUAL DE INTEGRACAO - CAMERA IA"
        print(f"Tentando inserir registro de teste...")
        
        sql_insert = "INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO) VALUES (:1, :2, :3, :4)"
        cursor.execute(sql_insert, [test_phone, test_msg, 0, 'G'])
        conn.commit()
        print("Inserção concluída e commit realizado.")
        
        # 2. Faz o SELECT solicitado pelo usuário
        print("\nConsultando a tabela DIZIMO.MENSAGENS (últimos 5 registros):")
        sql_select = "SELECT TELEFONE, TEXTO, STATUS, DATA_CADASTRO FROM (SELECT * FROM DIZIMO.MENSAGENS ORDER BY DATA_CADASTRO DESC) WHERE ROWNUM <= 5"
        cursor.execute(sql_select)
        
        rows = cursor.fetchall()
        if not rows:
            print("Nenhum registro encontrado na tabela.")
        for row in rows:
            print(f"Fone: {row[0]} | Msg: {row[1]} | Status: {row[2]} | Data: {row[3]}")
            
        cursor.close()
        conn.close()
        print("\nTeste finalizado.")
        
    except Exception as e:
        print(f"\nERRO NO TESTE: {e}")

if __name__ == "__main__":
    test_db()
